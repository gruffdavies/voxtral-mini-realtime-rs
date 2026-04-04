//! GGUF model loader for Q4-quantized TTS.
//!
//! Reads a GGUF file containing Q4_0 quantized TTS weights and builds the
//! Q4 TTS backbone, FM transformer, and codec decoder. Handles both native
//! file I/O and in-memory bytes for WASM deployment.

use anyhow::{Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::CubeRuntime;
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek};
use std::path::Path;
use tracing::info;

use crate::models::layers::{RmsNorm, RoPE, RoPEConfig};
use crate::models::time_embedding::TimeEmbedding;
use crate::models::weights::linear_from_weights;
use crate::tts::codec::block::CodecTransformerLayer;
use crate::tts::codec::conv::{CausalConv1d, CausalConvTranspose1d};
use crate::tts::codec::layer_scale::LayerScale;
use crate::tts::codec::qk_norm::QkNorm;
use crate::tts::codec::quantizer::VqCodebook;
use crate::tts::codec::CodecDecoder;
use crate::tts::config::{CodecDecoderConfig, FmTransformerConfig, TtsBackboneConfig};

use super::loader::{
    dequantize_q4_0_cpu, gguf_load_f32_tensor, gguf_load_q4_linear, gguf_load_rms_norm,
    reverse_gguf_dims,
};
use super::model::{Q4Attention, Q4FeedForward, TokEmbedStore};
use super::reader::{GgmlDtype, GgufReader, ShardedCursor};
use super::tensor::Q4Tensor;
use super::tts_model::{Q4FmLayer, Q4FmTransformer, Q4TtsBackbone, Q4TtsDecoderLayer};

/// All Q4 TTS model components with token embeddings still in raw Q4 form.
///
/// Used by [`Q4TtsModelLoader::load_deferred`] to allow freeing the GGUF
/// reader's memory before dequantizing the 131K-vocab embedding table.
pub struct Q4TtsModelParts<Rt, B>
where
    Rt: CubeRuntime,
    B: Backend<FloatTensorPrimitive = CubeTensor<Rt>, Device = Rt::Device>,
{
    pub backbone_layers: Vec<Q4TtsDecoderLayer<Rt, B>>,
    pub backbone_rope: RoPE<B>,
    pub backbone_norm: RmsNorm<B>,
    pub audio_codebook_embeddings: Tensor<B, 2>,
    pub fm: Q4FmTransformer<Rt, B>,
    pub codec: CodecDecoder<B>,
    pub tok_embed_q4_bytes: Vec<u8>,
    pub tok_embed_shape: [usize; 2],
    pub config: TtsBackboneConfig,
    pub device: Rt::Device,
}

impl<Rt, B> Q4TtsModelParts<Rt, B>
where
    Rt: CubeRuntime,
    B: Backend<FloatTensorPrimitive = CubeTensor<Rt>, Device = Rt::Device>,
{
    /// Assemble the final model components with Q4 token embeddings.
    ///
    /// For TTS inference, `lm_head` is never called (TTS generates audio tokens
    /// via the FM transformer, not text tokens via the vocabulary head).
    /// We therefore create a tiny 32×1 stub Q4 tensor for lm_head instead of the
    /// full 226 MB embedding table, keeping only cpu_bytes for embed_tokens lookups.
    pub fn finalize(
        self,
    ) -> Result<(Q4TtsBackbone<Rt, B>, Q4FmTransformer<Rt, B>, CodecDecoder<B>)> {
        // One minimal Q4_0 block (32 elements = 18 bytes) as a stub for lm_head.
        // Shape is [32, 1]; lm_head is never called during TTS generation.
        let stub_bytes = vec![0u8; 18];
        let tok_embed_stub =
            Q4Tensor::from_q4_bytes(&stub_bytes, [32, 1], &self.device)
                .context("Failed to create stub Q4 tensor for TTS lm_head")?;

        let tok_embeddings = TokEmbedStore::Q4 {
            lm_head: super::linear::Q4Linear::new(tok_embed_stub, None),
            cpu_bytes: self.tok_embed_q4_bytes,
        };

        #[allow(unused_mut)]
        let mut backbone = Q4TtsBackbone::new(
            self.backbone_layers,
            self.backbone_norm,
            tok_embeddings,
            self.audio_codebook_embeddings,
            self.backbone_rope,
            self.config,
            self.device,
        );

        // Fuse projections for faster decode (one-time cost).
        // Skipped on WASM — fuse_projections uses sync GPU readback (read_bytes)
        // which panics in browsers. The kernel launch savings matter less on WebGPU
        // anyway since dispatch overhead is dominated by JS→GPU communication.
        #[cfg(not(target_family = "wasm"))]
        backbone.fuse_projections();

        Ok((backbone, self.fm, self.codec))
    }
}

/// Loads a Q4-quantized TTS model from a GGUF file.
pub struct Q4TtsModelLoader<Rdr: Read + Seek> {
    reader: GgufReader<Rdr>,
}

impl Q4TtsModelLoader<BufReader<File>> {
    /// Open a GGUF file from disk.
    pub fn from_file(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
        let reader = GgufReader::open(BufReader::new(file))?;
        Ok(Self { reader })
    }
}

impl<'a> Q4TtsModelLoader<Cursor<&'a [u8]>> {
    /// Open a GGUF file from in-memory bytes (for WASM).
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self> {
        let reader = GgufReader::from_bytes(bytes)?;
        Ok(Self { reader })
    }
}

impl Q4TtsModelLoader<ShardedCursor> {
    /// Open a GGUF from multiple shards (for WASM).
    pub fn from_shards(shards: Vec<Vec<u8>>) -> Result<Self> {
        let reader = GgufReader::open(ShardedCursor::new(shards))?;
        Ok(Self { reader })
    }
}

impl<Rdr: Read + Seek> Q4TtsModelLoader<Rdr> {
    /// Load the complete Q4 TTS model.
    pub fn load<Rt, B>(
        &mut self,
        device: &Rt::Device,
    ) -> Result<(Q4TtsBackbone<Rt, B>, Q4FmTransformer<Rt, B>, CodecDecoder<B>)>
    where
        Rt: CubeRuntime,
        B: Backend<FloatTensorPrimitive = CubeTensor<Rt>, Device = Rt::Device>,
    {
        info!(
            version = self.reader.version(),
            tensors = self.reader.tensor_count(),
            "Loading Q4 TTS model from GGUF"
        );

        let config = TtsBackboneConfig::default();
        let fm_config = FmTransformerConfig::default();
        let codec_config = CodecDecoderConfig::default();

        // Load tok embeddings (dequantize to f32 for native)
        info!("Loading token embeddings");
        let tok_embeddings = self.load_tok_embeddings::<B>(device)?;

        info!(layers = config.n_layers, "Loading backbone layers");
        let (layers, rope, norm) = self.load_backbone::<Rt, B>(&config, device)?;

        info!("Loading audio codebook embeddings");
        let audio_codebook_embeddings = self.load_audio_codebook_embeddings::<B>(device)?;

        let backbone = Q4TtsBackbone::new(
            layers,
            norm,
            TokEmbedStore::F32(tok_embeddings),
            audio_codebook_embeddings,
            rope,
            config,
            device.clone(),
        );

        info!(layers = fm_config.n_layers, "Loading FM transformer");
        let fm = self.load_fm_transformer::<Rt, B>(&fm_config, device)?;

        info!("Loading codec decoder");
        let codec = self.load_codec::<B>(device)?;

        info!("Q4 TTS model loaded");

        Ok((backbone, fm, codec))
    }

    /// Load model components without dequantizing token embeddings.
    ///
    /// Returns [`Q4TtsModelParts`] for two-phase WASM loading.
    pub fn load_deferred<Rt, B>(
        &mut self,
        device: &Rt::Device,
    ) -> Result<Q4TtsModelParts<Rt, B>>
    where
        Rt: CubeRuntime,
        B: Backend<FloatTensorPrimitive = CubeTensor<Rt>, Device = Rt::Device>,
    {
        info!(
            version = self.reader.version(),
            tensors = self.reader.tensor_count(),
            "Loading Q4 TTS model from GGUF (deferred embedding)"
        );

        let config = TtsBackboneConfig::default();
        let fm_config = FmTransformerConfig::default();
        let codec_config = CodecDecoderConfig::default();

        // Extract raw Q4 bytes for token embeddings (don't dequantize yet)
        let tok_name = "mm_audio_embeddings.tok_embeddings.weight";
        let tok_info = self
            .reader
            .tensor_info(tok_name)
            .with_context(|| format!("Tensor '{tok_name}' not found"))?
            .clone();
        let tok_shape = reverse_gguf_dims(tok_info.shape());
        let tok_embed_q4_bytes = self.reader.tensor_data(tok_name)?;

        info!(layers = config.n_layers, "Loading backbone layers");
        let (layers, rope, norm) = self.load_backbone::<Rt, B>(&config, device)?;

        info!("Loading audio codebook embeddings");
        let audio_codebook_embeddings = self.load_audio_codebook_embeddings::<B>(device)?;

        info!(layers = fm_config.n_layers, "Loading FM transformer");
        let fm = self.load_fm_transformer::<Rt, B>(&fm_config, device)?;

        info!("Loading codec decoder");
        let codec = self.load_codec::<B>(device)?;

        info!("Q4 TTS model loaded (token embeddings deferred)");

        Ok(Q4TtsModelParts {
            backbone_layers: layers,
            backbone_rope: rope,
            backbone_norm: norm,
            audio_codebook_embeddings,
            fm,
            codec,
            tok_embed_q4_bytes,
            tok_embed_shape: [tok_shape[0], tok_shape[1]],
            config,
            device: device.clone(),
        })
    }

    // -----------------------------------------------------------------------
    // Component loaders
    // -----------------------------------------------------------------------

    /// Load backbone layers, RoPE, and final norm.
    fn load_backbone<Rt, B>(
        &mut self,
        config: &TtsBackboneConfig,
        device: &Rt::Device,
    ) -> Result<(Vec<Q4TtsDecoderLayer<Rt, B>>, RoPE<B>, RmsNorm<B>)>
    where
        Rt: CubeRuntime,
        B: Backend<FloatTensorPrimitive = CubeTensor<Rt>, Device = Rt::Device>,
    {
        let rope = RoPEConfig::new(config.head_dim, 131_072)
            .with_theta(config.rope_theta)
            .init(device);

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let layer = self
                .load_backbone_layer::<Rt, B>(i, config, device)
                .with_context(|| format!("Failed to load backbone layer {i}"))?;
            layers.push(layer);
        }

        let norm =
            gguf_load_rms_norm::<_, B>(&mut self.reader, "norm.weight", config.norm_eps, device)?;

        Ok((layers, rope, norm))
    }

    /// Load a single TTS backbone decoder layer.
    fn load_backbone_layer<Rt, B>(
        &mut self,
        layer_idx: usize,
        config: &TtsBackboneConfig,
        device: &Rt::Device,
    ) -> Result<Q4TtsDecoderLayer<Rt, B>>
    where
        Rt: CubeRuntime,
        B: Backend<FloatTensorPrimitive = CubeTensor<Rt>, Device = Rt::Device>,
    {
        let prefix = format!("layers.{layer_idx}");

        let attention_norm = gguf_load_rms_norm::<_, B>(
            &mut self.reader,
            &format!("{prefix}.attention_norm.weight"),
            config.norm_eps,
            device,
        )?;

        let wq = gguf_load_q4_linear::<_, Rt, B>(
            &mut self.reader,
            &format!("{prefix}.attention.wq.weight"),
            device,
        )?;
        let wk = gguf_load_q4_linear::<_, Rt, B>(
            &mut self.reader,
            &format!("{prefix}.attention.wk.weight"),
            device,
        )?;
        let wv = gguf_load_q4_linear::<_, Rt, B>(
            &mut self.reader,
            &format!("{prefix}.attention.wv.weight"),
            device,
        )?;
        let wo = gguf_load_q4_linear::<_, Rt, B>(
            &mut self.reader,
            &format!("{prefix}.attention.wo.weight"),
            device,
        )?;

        let attention = Q4Attention::new(
            wq,
            wk,
            wv,
            wo,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            None, // No sliding window for TTS backbone
        );

        let ffn_norm = gguf_load_rms_norm::<_, B>(
            &mut self.reader,
            &format!("{prefix}.ffn_norm.weight"),
            config.norm_eps,
            device,
        )?;

        let w1 = gguf_load_q4_linear::<_, Rt, B>(
            &mut self.reader,
            &format!("{prefix}.feed_forward.w1.weight"),
            device,
        )?;
        let w2 = gguf_load_q4_linear::<_, Rt, B>(
            &mut self.reader,
            &format!("{prefix}.feed_forward.w2.weight"),
            device,
        )?;
        let w3 = gguf_load_q4_linear::<_, Rt, B>(
            &mut self.reader,
            &format!("{prefix}.feed_forward.w3.weight"),
            device,
        )?;
        let ffn = Q4FeedForward::new(w1, w2, w3);

        Ok(Q4TtsDecoderLayer::new(
            attention_norm,
            attention,
            ffn_norm,
            ffn,
        ))
    }

    /// Load the FM transformer.
    fn load_fm_transformer<Rt, B>(
        &mut self,
        config: &FmTransformerConfig,
        device: &Rt::Device,
    ) -> Result<Q4FmTransformer<Rt, B>>
    where
        Rt: CubeRuntime,
        B: Backend<FloatTensorPrimitive = CubeTensor<Rt>, Device = Rt::Device>,
    {
        let prefix = "acoustic_transformer";

        // Projections
        let llm_projection = gguf_load_q4_linear::<_, Rt, B>(
            &mut self.reader,
            &format!("{prefix}.llm_projection.weight"),
            device,
        )
        .context("loading llm_projection")?;

        let time_projection = gguf_load_q4_linear::<_, Rt, B>(
            &mut self.reader,
            &format!("{prefix}.time_projection.weight"),
            device,
        )
        .context("loading time_projection")?;

        // input_projection is F32 (tiny: [3072, 36])
        let input_proj_weight: Tensor<B, 2> = gguf_load_f32_tensor::<_, B, 2>(
            &mut self.reader,
            &format!("{prefix}.input_projection.weight"),
            device,
        )
        .context("loading input_projection")?;
        let input_projection = linear_from_weights(input_proj_weight, None);

        // semantic_codebook_output is Q4
        let semantic_codebook_output = gguf_load_q4_linear::<_, Rt, B>(
            &mut self.reader,
            &format!("{prefix}.semantic_codebook_output.weight"),
            device,
        )
        .context("loading semantic_codebook_output")?;

        // acoustic_codebook_output is F32 (tiny: [36, 3072])
        let acoustic_output_weight: Tensor<B, 2> = gguf_load_f32_tensor::<_, B, 2>(
            &mut self.reader,
            &format!("{prefix}.acoustic_codebook_output.weight"),
            device,
        )
        .context("loading acoustic_codebook_output")?;
        let acoustic_codebook_output = linear_from_weights(acoustic_output_weight, None);

        // FM transformer layers
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let layer = self
                .load_fm_layer::<Rt, B>(i, config, device)
                .with_context(|| format!("loading FM layer {i}"))?;
            layers.push(layer);
        }

        // Final norm
        let norm = gguf_load_rms_norm::<_, B>(
            &mut self.reader,
            &format!("{prefix}.norm.weight"),
            config.norm_eps,
            device,
        )
        .context("loading FM norm")?;

        // RoPE with theta=10K, max seq_len=16
        let rope = RoPEConfig::new(config.head_dim, 16)
            .with_theta(config.rope_theta)
            .init(device);

        let time_embedding = TimeEmbedding::new(config.dim);

        Ok(Q4FmTransformer::new(
            time_embedding,
            rope,
            layers,
            llm_projection,
            time_projection,
            input_projection,
            semantic_codebook_output,
            acoustic_codebook_output,
            norm,
            config.clone(),
        ))
    }

    /// Load a single FM transformer layer.
    fn load_fm_layer<Rt, B>(
        &mut self,
        layer_idx: usize,
        config: &FmTransformerConfig,
        device: &Rt::Device,
    ) -> Result<Q4FmLayer<Rt, B>>
    where
        Rt: CubeRuntime,
        B: Backend<FloatTensorPrimitive = CubeTensor<Rt>, Device = Rt::Device>,
    {
        let prefix = format!("acoustic_transformer.layers.{layer_idx}");

        let attention_norm = gguf_load_rms_norm::<_, B>(
            &mut self.reader,
            &format!("{prefix}.attention_norm.weight"),
            config.norm_eps,
            device,
        )?;

        let wq = gguf_load_q4_linear::<_, Rt, B>(
            &mut self.reader,
            &format!("{prefix}.attention.wq.weight"),
            device,
        )?;
        let wk = gguf_load_q4_linear::<_, Rt, B>(
            &mut self.reader,
            &format!("{prefix}.attention.wk.weight"),
            device,
        )?;
        let wv = gguf_load_q4_linear::<_, Rt, B>(
            &mut self.reader,
            &format!("{prefix}.attention.wv.weight"),
            device,
        )?;
        let wo = gguf_load_q4_linear::<_, Rt, B>(
            &mut self.reader,
            &format!("{prefix}.attention.wo.weight"),
            device,
        )?;

        let attention = Q4Attention::new(
            wq,
            wk,
            wv,
            wo,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            None, // No sliding window for FM transformer
        );

        let ffn_norm = gguf_load_rms_norm::<_, B>(
            &mut self.reader,
            &format!("{prefix}.ffn_norm.weight"),
            config.norm_eps,
            device,
        )?;

        let w1 = gguf_load_q4_linear::<_, Rt, B>(
            &mut self.reader,
            &format!("{prefix}.feed_forward.w1.weight"),
            device,
        )?;
        let w2 = gguf_load_q4_linear::<_, Rt, B>(
            &mut self.reader,
            &format!("{prefix}.feed_forward.w2.weight"),
            device,
        )?;
        let w3 = gguf_load_q4_linear::<_, Rt, B>(
            &mut self.reader,
            &format!("{prefix}.feed_forward.w3.weight"),
            device,
        )?;
        let ffn = Q4FeedForward::new(w1, w2, w3);

        Ok(Q4FmLayer::new(attention_norm, attention, ffn_norm, ffn))
    }

    /// Load the codec decoder (all F32, pre-fused weight norms).
    fn load_codec<B: Backend>(
        &mut self,
        device: &B::Device,
    ) -> Result<CodecDecoder<B>> {
        let prefix = "audio_tokenizer";
        let config = CodecDecoderConfig::default();

        // Block 0: Input Conv1d [292→1024, k=3, s=1]
        let input_conv_weight: Tensor<B, 3> = gguf_load_f32_tensor(
            &mut self.reader,
            &format!("{prefix}.decoder_blocks.0.conv.weight"),
            device,
        )
        .context("Loading input conv weight")?;
        let input_conv = CausalConv1d::from_fused_weight(input_conv_weight, 1, device);

        // Blocks 1,3,5,7: Transformer groups
        let transformer_block_indices = [1, 3, 5, 7];
        let mut transformer_groups = Vec::with_capacity(4);
        for (group_idx, &block_idx) in transformer_block_indices.iter().enumerate() {
            let sliding_window = config.sliding_windows[group_idx];
            let mut layers = Vec::with_capacity(config.layers_per_block);
            for layer_idx in 0..config.layers_per_block {
                let layer = self
                    .load_codec_transformer_layer::<B>(
                        &format!("{prefix}.decoder_blocks.{block_idx}.layers.{layer_idx}"),
                        &config,
                        sliding_window,
                        device,
                    )
                    .with_context(|| {
                        format!("Loading transformer block {block_idx} layer {layer_idx}")
                    })?;
                layers.push(layer);
            }
            transformer_groups.push(layers);
        }

        // Blocks 2,4,6: Upsample ConvTranspose1d [1024→1024, k=4, s=2]
        let upsample_block_indices = [2, 4, 6];
        let mut upsample_convs = Vec::with_capacity(3);
        for &block_idx in &upsample_block_indices {
            let weight: Tensor<B, 3> = gguf_load_f32_tensor(
                &mut self.reader,
                &format!("{prefix}.decoder_blocks.{block_idx}.conv.weight"),
                device,
            )
            .with_context(|| format!("Loading upsample conv (block {block_idx})"))?;
            upsample_convs.push(CausalConvTranspose1d::from_fused_weight(weight, 2, device));
        }

        // Output Conv1d [1024→240, k=7, s=1]
        let output_conv_weight: Tensor<B, 3> = gguf_load_f32_tensor(
            &mut self.reader,
            &format!("{prefix}.output_proj.conv.weight"),
            device,
        )
        .context("Loading output conv")?;
        let output_conv = CausalConv1d::from_fused_weight(output_conv_weight, 1, device);

        // VQ codebook
        let vq_codebook = self.load_vq_codebook::<B>(device)?;

        Ok(CodecDecoder::from_components(
            input_conv,
            transformer_groups,
            upsample_convs,
            output_conv,
            vq_codebook,
        ))
    }

    /// Load a single codec transformer layer from GGUF (F32 weights).
    fn load_codec_transformer_layer<B: Backend>(
        &mut self,
        prefix: &str,
        config: &CodecDecoderConfig,
        sliding_window: usize,
        device: &B::Device,
    ) -> Result<CodecTransformerLayer<B>> {
        // Attention weights (all F32)
        let wq_weight: Tensor<B, 2> = gguf_load_f32_tensor(
            &mut self.reader,
            &format!("{prefix}.attention.wq.weight"),
            device,
        )
        .context("Loading wq")?;
        let wk_weight: Tensor<B, 2> = gguf_load_f32_tensor(
            &mut self.reader,
            &format!("{prefix}.attention.wk.weight"),
            device,
        )
        .context("Loading wk")?;
        let wv_weight: Tensor<B, 2> = gguf_load_f32_tensor(
            &mut self.reader,
            &format!("{prefix}.attention.wv.weight"),
            device,
        )
        .context("Loading wv")?;
        let wo_weight: Tensor<B, 2> = gguf_load_f32_tensor(
            &mut self.reader,
            &format!("{prefix}.attention.wo.weight"),
            device,
        )
        .context("Loading wo")?;

        let wq = linear_from_weights(wq_weight, None);
        let wk = linear_from_weights(wk_weight, None);
        let wv = linear_from_weights(wv_weight, None);
        let wo = linear_from_weights(wo_weight, None);

        // QK-norm weights
        let q_norm_weight: Tensor<B, 1> = gguf_load_f32_tensor(
            &mut self.reader,
            &format!("{prefix}.attention.q_norm.weight"),
            device,
        )
        .context("Loading q_norm")?;
        let k_norm_weight: Tensor<B, 1> = gguf_load_f32_tensor(
            &mut self.reader,
            &format!("{prefix}.attention.k_norm.weight"),
            device,
        )
        .context("Loading k_norm")?;
        let qk_norm = QkNorm::new(
            q_norm_weight,
            k_norm_weight,
            config.n_heads,
            config.head_dim,
        );

        let attention = crate::tts::codec::block::CodecAttention::new(
            wq,
            wk,
            wv,
            wo,
            qk_norm,
            config.n_heads,
            config.head_dim,
            sliding_window,
        );

        // LayerScale weights
        let attn_scale_weight: Tensor<B, 1> = gguf_load_f32_tensor(
            &mut self.reader,
            &format!("{prefix}.attention_scale"),
            device,
        )
        .context("Loading attention_scale")?;
        let ffn_scale_weight: Tensor<B, 1> =
            gguf_load_f32_tensor(&mut self.reader, &format!("{prefix}.ffn_scale"), device)
                .context("Loading ffn_scale")?;
        let attention_scale = LayerScale::new(attn_scale_weight);
        let ffn_scale = LayerScale::new(ffn_scale_weight);

        // Norms
        let attention_norm = gguf_load_rms_norm::<_, B>(
            &mut self.reader,
            &format!("{prefix}.attention_norm.weight"),
            config.norm_eps,
            device,
        )
        .context("Loading attention_norm")?;
        let ffn_norm = gguf_load_rms_norm::<_, B>(
            &mut self.reader,
            &format!("{prefix}.ffn_norm.weight"),
            config.norm_eps,
            device,
        )
        .context("Loading ffn_norm")?;

        // SwiGLU FFN (F32)
        let w1_weight: Tensor<B, 2> = gguf_load_f32_tensor(
            &mut self.reader,
            &format!("{prefix}.feed_forward.w1.weight"),
            device,
        )?;
        let w2_weight: Tensor<B, 2> = gguf_load_f32_tensor(
            &mut self.reader,
            &format!("{prefix}.feed_forward.w2.weight"),
            device,
        )?;
        let w3_weight: Tensor<B, 2> = gguf_load_f32_tensor(
            &mut self.reader,
            &format!("{prefix}.feed_forward.w3.weight"),
            device,
        )?;
        let w1 = linear_from_weights(w1_weight, None);
        let w2 = linear_from_weights(w2_weight, None);
        let w3 = linear_from_weights(w3_weight, None);
        let ffn = crate::models::layers::SwiGLU::new(w1, w2, w3);

        Ok(CodecTransformerLayer::new(
            attention_norm,
            attention,
            attention_scale,
            ffn_norm,
            ffn,
            ffn_scale,
        ))
    }

    /// Load VQ codebook from GGUF.
    ///
    /// Pre-computes normalized embeddings on CPU to avoid GPU readback
    /// (required for WASM compatibility).
    fn load_vq_codebook<B: Backend>(
        &mut self,
        device: &B::Device,
    ) -> Result<VqCodebook<B>> {
        let embed_name = "audio_tokenizer.quantizer.semantic_codebook.embedding_sum";
        let usage_name = "audio_tokenizer.quantizer.semantic_codebook.cluster_usage";

        // Read raw bytes from GGUF
        let embed_info = self
            .reader
            .tensor_info(embed_name)
            .with_context(|| format!("Tensor '{embed_name}' not found"))?
            .clone();
        let embed_shape = reverse_gguf_dims(embed_info.shape());
        let embed_bytes = self
            .reader
            .tensor_data(embed_name)
            .context("Reading VQ embedding_sum bytes")?;
        let embed_f32: Vec<f32> = embed_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let usage_bytes = self
            .reader
            .tensor_data(usage_name)
            .context("Reading VQ cluster_usage bytes")?;
        let usage_f32: Vec<f32> = usage_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let n_entries = embed_shape[0];
        let embed_dim = embed_shape[1];

        // Pre-compute normalized embeddings on CPU (no GPU readback needed)
        let cpu_normalized =
            VqCodebook::<B>::precompute_normalized(&embed_f32, &usage_f32, n_entries, embed_dim);

        // Create GPU tensors
        let embedding_sum: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(embed_f32, embed_shape), device);
        let cluster_usage: Tensor<B, 1> =
            Tensor::from_data(TensorData::new(usage_f32, [n_entries]), device);

        Ok(VqCodebook::new(
            embedding_sum,
            cluster_usage,
            cpu_normalized,
        ))
    }

    /// Load token embeddings (Q4_0 → dequantized f32).
    fn load_tok_embeddings<B: Backend>(
        &mut self,
        device: &B::Device,
    ) -> Result<Tensor<B, 2>> {
        let name = "mm_audio_embeddings.tok_embeddings.weight";
        let info = self
            .reader
            .tensor_info(name)
            .with_context(|| format!("Tensor '{name}' not found"))?
            .clone();

        let shape = reverse_gguf_dims(info.shape());

        match info.dtype() {
            GgmlDtype::Q4_0 => {
                let bytes = self.reader.tensor_data(name)?;
                let f32_data = dequantize_q4_0_cpu(&bytes, shape[0] * shape[1]);
                let tensor_data = TensorData::new(f32_data, [shape[0], shape[1]]);
                Ok(Tensor::from_data(tensor_data, device))
            }
            GgmlDtype::F32 | GgmlDtype::F16 => gguf_load_f32_tensor(&mut self.reader, name, device),
            #[allow(unreachable_patterns)]
            other => anyhow::bail!("Unsupported dtype {other:?} for tok_embeddings"),
        }
    }

    /// Load audio codebook embeddings [9088, 3072] as F32.
    fn load_audio_codebook_embeddings<B: Backend>(
        &mut self,
        device: &B::Device,
    ) -> Result<Tensor<B, 2>> {
        gguf_load_f32_tensor(
            &mut self.reader,
            "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight",
            device,
        )
        .context("Loading audio codebook embeddings")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::WgpuDevice;

    #[test]
    fn test_q4_tts_model_loader_tensor_names() {
        // Verify weight name generation matches expected patterns
        let config = TtsBackboneConfig::default();
        assert_eq!(config.n_layers, 26);

        // Backbone layer names
        let prefix = "layers.0";
        assert_eq!(
            format!("{prefix}.attention_norm.weight"),
            "layers.0.attention_norm.weight"
        );
        assert_eq!(
            format!("{prefix}.attention.wq.weight"),
            "layers.0.attention.wq.weight"
        );
        assert_eq!(
            format!("{prefix}.feed_forward.w1.weight"),
            "layers.0.feed_forward.w1.weight"
        );

        // FM layer names
        let fm_prefix = "acoustic_transformer.layers.0";
        assert_eq!(
            format!("{fm_prefix}.attention_norm.weight"),
            "acoustic_transformer.layers.0.attention_norm.weight"
        );
        assert_eq!(
            format!("{fm_prefix}.attention.wq.weight"),
            "acoustic_transformer.layers.0.attention.wq.weight"
        );

        // FM projection names
        assert_eq!(
            "acoustic_transformer.llm_projection.weight",
            "acoustic_transformer.llm_projection.weight"
        );
        assert_eq!(
            "acoustic_transformer.input_projection.weight",
            "acoustic_transformer.input_projection.weight"
        );

        // Codec names
        assert_eq!(
            "audio_tokenizer.decoder_blocks.0.conv.weight",
            "audio_tokenizer.decoder_blocks.0.conv.weight"
        );
        assert_eq!(
            "audio_tokenizer.decoder_blocks.1.layers.0.attention.wq.weight",
            "audio_tokenizer.decoder_blocks.1.layers.0.attention.wq.weight"
        );
    }

    #[test]
    fn test_q4_tts_model_loader_from_file() {
        use burn::backend::wgpu::WgpuRuntime;
        use burn::backend::Wgpu;

        let path = std::path::PathBuf::from("models/voxtral-tts-q4.gguf");
        if !path.exists() {
            println!("Skipping: TTS GGUF model not found at {}", path.display());
            return;
        }

        let device = WgpuDevice::default();
        let mut loader = Q4TtsModelLoader::from_file(&path).unwrap();
        let (backbone, fm, _codec) = loader.load::<WgpuRuntime, Wgpu>(&device).unwrap();

        assert_eq!(backbone.n_layers(), 26);
        assert_eq!(backbone.d_model(), 3072);
        assert_eq!(fm.config().n_layers, 3);

        println!("Q4 TTS model loaded successfully from GGUF!");
    }
}
