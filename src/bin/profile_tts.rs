//! CUDA TTS profiling harness for Nsight Systems.
//!
//! Runs one warmup pass (to ensure autotune cache is populated), then one
//! profiled pass annotated with NVTX ranges.
//!
//! # Usage
//!
//! ```bash
//! # Build
//! cargo build --release --features "wgpu,cli,native-tokenizer,cuda" --bin profile-tts
//!
//! # Profile with Nsight Systems (warm autotune cache recommended)
//! CUBECL_AUTOTUNE_LEVEL=minimal \
//! nsys profile \
//!   --trace=cuda,nvtx \
//!   --output profile_out \
//!   ./target/release/profile-tts \
//!     --gguf models/voxtral-tts-q4.gguf \
//!     --text "The quick brown fox jumps over the lazy dog"
//!
//! # Then open profile_out.nsys-rep in Nsight Systems GUI.
//! # Look for the "PROFILED_RUN" NVTX range on the CPU timeline.
//! # Zoom in to one "tts_frame_N" range to see the CPU/GPU interleave.
//! ```
//!
//! # What to look for
//!
//! If **many tiny GPU kernels with CPU gaps** dominate: kernel launch overhead
//! is the bottleneck → focus on CUDA Graphs / kernel fusion.
//!
//! If **a few large GPU kernels** dominate: those specific ops are the bottleneck
//! → profile those kernels (attention matmuls, q4_matmul) for occupancy.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use burn::tensor::Tensor;
use burn_cubecl::CubeBackend;
use clap::Parser;
use cubecl::cuda::CudaRuntime;
use tracing::info;
use voxtral_mini_realtime::{
    gguf::Q4TtsModelLoader,
    nvtx_range,
    tts::{
        config::{AudioCodebookLayout, TtsSpecialTokens},
        embeddings::AudioCodebookEmbeddings,
        voice::load_voice_from_bytes,
    },
};
use voxtral_mini_realtime::tokenizer::TekkenEncoder;

type CudaBackend = CubeBackend<CudaRuntime, f32, i32, u8>;

#[derive(Parser)]
#[command(name = "profile-tts", about = "CUDA TTS profiling harness for Nsight Systems")]
struct Args {
    /// Path to Q4 GGUF TTS model
    #[arg(long)]
    gguf: String,

    /// Text to synthesize
    #[arg(long, default_value = "The quick brown fox jumps over the lazy dog. \
        She sells seashells by the seashore.")]
    text: String,

    /// Voice preset name
    #[arg(long, default_value = "casual_female")]
    voice: String,

    /// Path to voice embeddings directory
    #[arg(long, default_value = "models/voxtral-tts/voice_embedding")]
    voices_dir: String,

    /// Path to Tekken tokenizer JSON
    #[arg(long)]
    tokenizer: Option<String>,

    /// Number of Euler ODE steps
    #[arg(long, default_value_t = 3)]
    euler_steps: usize,

    /// Number of warmup runs before the profiled run
    #[arg(long, default_value_t = 1)]
    warmup_runs: usize,

    /// Maximum audio frames to generate
    #[arg(long, default_value_t = 500)]
    max_frames: usize,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("info".parse().unwrap()),
        )
        .init();

    let args = Args::parse();
    let device = cubecl::cuda::CudaDevice::default();
    info!("CUDA device: {device:?}");

    // -------------------------------------------------------------------------
    // Load tokenizer
    // -------------------------------------------------------------------------
    let tokenizer_path = match &args.tokenizer {
        Some(p) => PathBuf::from(p),
        None => {
            let gguf_dir = PathBuf::from(&args.gguf)
                .parent()
                .unwrap_or(&PathBuf::from("."))
                .to_path_buf();
            [
                gguf_dir.join("tekken.json"),
                PathBuf::from("models/voxtral-tts/tekken.json"),
                PathBuf::from("models/voxtral/tekken.json"),
            ]
            .into_iter()
            .find(|p| p.exists())
            .ok_or_else(|| anyhow::anyhow!("Tokenizer not found. Provide --tokenizer"))?
        }
    };
    let encoder =
        TekkenEncoder::from_file(&tokenizer_path).context("Failed to load tokenizer")?;
    let token_ids: Vec<u32> = encoder.encode(&args.text);
    info!(tokens = token_ids.len(), text = %args.text, "Text tokenized");

    // -------------------------------------------------------------------------
    // Load model
    // -------------------------------------------------------------------------
    let path = PathBuf::from(&args.gguf);
    if !path.exists() {
        bail!("GGUF model not found at {}", path.display());
    }
    let load_start = std::time::Instant::now();
    info!("Loading Q4 TTS model from {}", path.display());
    let mut loader = Q4TtsModelLoader::from_file(&path).context("Failed to open GGUF")?;
    let (backbone, mut fm, _codec) = loader
        .load_deferred::<CudaRuntime, CudaBackend>(&device)
        .context("Failed to load Q4 model")?
        .finalize()
        .context("Failed to finalize Q4 model")?;
    fm.set_euler_steps(args.euler_steps);
    info!(elapsed_ms = load_start.elapsed().as_millis() as u64, "Model loaded");

    // -------------------------------------------------------------------------
    // Load voice
    // -------------------------------------------------------------------------
    let voice_path = PathBuf::from(&args.voices_dir).join(format!("{}.safetensors", args.voice));
    if !voice_path.exists() {
        bail!(
            "Voice '{}' not found at {}",
            args.voice,
            voice_path.display()
        );
    }
    let voice_bytes = std::fs::read(&voice_path)?;
    let voice_embed: Tensor<CudaBackend, 2> =
        load_voice_from_bytes(&voice_bytes, 3072, &device).context("Failed to load voice")?;

    // -------------------------------------------------------------------------
    // Build input sequence (reused for warmup + profiled run)
    // -------------------------------------------------------------------------
    let special = TtsSpecialTokens::default();
    let bos = backbone.embed_tokens_from_ids(&[special.bos_token_id as i32], 1, 1);
    let begin_audio =
        backbone.embed_tokens_from_ids(&[special.begin_audio_token_id as i32], 1, 1);
    let next_audio_text =
        backbone.embed_tokens_from_ids(&[special.next_audio_text_token_id as i32], 1, 1);
    let repeat_audio_text =
        backbone.embed_tokens_from_ids(&[special.repeat_audio_text_token_id as i32], 1, 1);
    let text_ids_i32: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
    let text_embeds = backbone.embed_tokens_from_ids(&text_ids_i32, 1, text_ids_i32.len());

    let codebook = AudioCodebookEmbeddings::new(
        backbone.audio_codebook_embeddings().clone(),
        AudioCodebookLayout::default(),
    );

    let build_input = || {
        Tensor::cat(
            vec![
                bos.clone(),
                begin_audio.clone(),
                voice_embed.clone().unsqueeze_dim::<3>(0),
                next_audio_text.clone(),
                text_embeds.clone(),
                repeat_audio_text.clone(),
                begin_audio.clone(),
            ],
            1,
        )
    };

    // -------------------------------------------------------------------------
    // Warmup runs — populate autotune cache, JIT any remaining kernels
    // -------------------------------------------------------------------------
    for i in 0..args.warmup_runs {
        info!("Warmup run {}/{}", i + 1, args.warmup_runs);
        let input = build_input();
        pollster::block_on(backbone.generate_async(input, &fm, &codebook, args.max_frames))
            .map_err(|e| anyhow::anyhow!("Warmup generation failed: {e}"))?;
    }
    info!("Warmup complete — autotune cache should be warm");

    // -------------------------------------------------------------------------
    // Profiled run — annotated with NVTX ranges
    // -------------------------------------------------------------------------
    info!("=== PROFILED RUN STARTING ===");
    let input = build_input();

    let prof_start = std::time::Instant::now();
    let frames = {
        nvtx_range!("PROFILED_RUN");
        pollster::block_on(backbone.generate_async(input, &fm, &codebook, args.max_frames))
            .map_err(|e| anyhow::anyhow!("Profiled generation failed: {e}"))?
    };
    let elapsed = prof_start.elapsed();

    let n_frames = frames.len();
    let audio_sec = n_frames as f64 * 0.08; // ~80ms per frame at 12.5 Hz
    let rtf = elapsed.as_secs_f64() / audio_sec;

    info!(
        frames = n_frames,
        elapsed_ms = elapsed.as_millis() as u64,
        audio_sec = format!("{audio_sec:.2}"),
        rtf = format!("{rtf:.2}"),
        "=== PROFILED RUN COMPLETE ==="
    );

    Ok(())
}
