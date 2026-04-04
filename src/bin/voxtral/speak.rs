//! `voxtral speak` subcommand — text-to-speech.

use anyhow::{bail, Context, Result};
use burn::backend::Wgpu;
use burn::tensor::Tensor;
use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use std::time::Instant;
use tracing::info;

use voxtral_mini_realtime::audio::AudioBuffer;
use voxtral_mini_realtime::tokenizer::TekkenEncoder;

type Backend = Wgpu;

/// Compute device selection for Q4 GGUF inference.
#[derive(Debug, Clone, ValueEnum, Default)]
pub enum DeviceArg {
    /// WebGPU/Vulkan via wgpu (default; falls back to llvmpipe on WSL2).
    #[default]
    Wgpu,
    /// NVIDIA CUDA (requires cuda feature and CUDA toolkit).
    #[cfg(feature = "cuda")]
    Cuda,
}

#[derive(Parser)]
pub struct Args {
    /// Text to synthesize.
    #[arg(short, long, required_unless_present_any = ["list_voices", "token_ids"])]
    text: Option<String>,

    /// Pre-tokenized token IDs (comma-separated, bypasses tokenizer).
    #[arg(long, value_delimiter = ',')]
    token_ids: Option<Vec<u32>>,

    /// BF16 SafeTensors model directory.
    #[arg(
        short,
        long,
        default_value = "models/voxtral-tts",
        conflicts_with = "gguf"
    )]
    model: String,

    /// Q4 GGUF model file (use instead of --model for quantized inference).
    #[arg(long, conflicts_with = "model")]
    gguf: Option<String>,

    /// Voice preset name.
    #[arg(short, long, default_value = "casual_female")]
    voice: String,

    /// Voice embeddings directory (auto-discovered from --model dir for BF16).
    #[arg(long)]
    voices_dir: Option<String>,

    /// Output WAV file path.
    #[arg(short, long, default_value = "output.wav")]
    output: String,

    /// Tekken tokenizer JSON (auto-discovered from model dir).
    #[arg(long)]
    tokenizer: Option<String>,

    /// List available voice presets and exit.
    #[arg(long)]
    list_voices: bool,

    /// Maximum audio frames to generate.
    #[arg(long, default_value_t = 2000)]
    max_frames: usize,

    /// Euler ODE steps: 3=real-time, 4=balanced, 8=quality.
    #[arg(long, default_value_t = 4)]
    euler_steps: usize,

    /// Compute device for Q4 GGUF inference.
    #[arg(long, default_value = "wgpu")]
    device: DeviceArg,
}

pub fn run(args: Args) -> Result<()> {
    match args.device {
        #[cfg(feature = "cuda")]
        DeviceArg::Cuda => return run_cuda(args),
        _ => {}
    }
    run_wgpu(args)
}

#[cfg(feature = "cuda")]
fn run_cuda(args: Args) -> Result<()> {
    // Use CubeBackend<CudaRuntime, f32, i32> directly (non-fusion) so that
    // FloatTensorPrimitive = CubeTensor<CudaRuntime> matches our Q4 constraint.
    use burn_cubecl::CubeBackend;
    use cubecl::cuda::CudaRuntime;

    type CudaBackend = CubeBackend<CudaRuntime, f32, i32, u8>;

    let device = cubecl::cuda::CudaDevice::default();
    info!("CUDA device: {device:?}");

    // Resolve tokenizer
    let tokenizer_path = match &args.tokenizer {
        Some(p) => PathBuf::from(p),
        None => {
            if let Some(gguf) = &args.gguf {
                let gguf_dir = PathBuf::from(gguf)
                    .parent()
                    .unwrap_or(&PathBuf::from("."))
                    .to_path_buf();
                let candidates = [
                    gguf_dir.join("tekken.json"),
                    PathBuf::from("models/voxtral-tts/tekken.json"),
                    PathBuf::from("models/voxtral/tekken.json"),
                ];
                candidates
                    .into_iter()
                    .find(|p| p.exists())
                    .ok_or_else(|| anyhow::anyhow!("Tokenizer not found. Provide --tokenizer"))?
            } else {
                PathBuf::from(&args.model).join("tekken.json")
            }
        }
    };

    let token_ids: Vec<u32> = if let Some(ids) = &args.token_ids {
        ids.clone()
    } else if let Some(text) = &args.text {
        if !tokenizer_path.exists() {
            bail!("Tokenizer not found at {}", tokenizer_path.display());
        }
        let encoder =
            TekkenEncoder::from_file(&tokenizer_path).context("Failed to load tokenizer")?;
        encoder.encode(text)
    } else if !args.list_voices {
        bail!("--text or --token-ids required");
    } else {
        vec![]
    };

    if let Some(gguf_path) = &args.gguf {
        run_q4::<CudaRuntime, CudaBackend>(&args, gguf_path, &token_ids, &device)
    } else {
        bail!("CUDA inference requires --gguf (BF16 safetensors path not supported on CUDA)")
    }
}

fn run_wgpu(args: Args) -> Result<()> {
    let device = burn::backend::wgpu::WgpuDevice::default();

    // Resolve tokenizer
    let tokenizer_path = match &args.tokenizer {
        Some(p) => PathBuf::from(p),
        None => {
            if let Some(gguf) = &args.gguf {
                // Try alongside GGUF, then common locations
                let gguf_dir = PathBuf::from(gguf)
                    .parent()
                    .unwrap_or(&PathBuf::from("."))
                    .to_path_buf();
                let candidates = [
                    gguf_dir.join("tekken.json"),
                    PathBuf::from("models/voxtral-tts/tekken.json"),
                    PathBuf::from("models/voxtral/tekken.json"),
                ];
                candidates
                    .into_iter()
                    .find(|p| p.exists())
                    .ok_or_else(|| anyhow::anyhow!("Tokenizer not found. Provide --tokenizer"))?
            } else {
                PathBuf::from(&args.model).join("tekken.json")
            }
        }
    };

    // Get token IDs
    let token_ids: Vec<u32> = if let Some(ids) = &args.token_ids {
        ids.clone()
    } else if let Some(text) = &args.text {
        if !tokenizer_path.exists() {
            bail!("Tokenizer not found at {}", tokenizer_path.display());
        }
        let encoder =
            TekkenEncoder::from_file(&tokenizer_path).context("Failed to load tokenizer")?;
        encoder.encode(text)
    } else if !args.list_voices {
        bail!("--text or --token-ids required");
    } else {
        vec![]
    };

    if let Some(gguf_path) = &args.gguf {
        run_q4::<burn::backend::wgpu::WgpuRuntime, burn::backend::Wgpu>(&args, gguf_path, &token_ids, &device)
    } else {
        run_bf16(&args, &token_ids, &device)
    }
}

/// BF16 SafeTensors path.
fn run_bf16(
    args: &Args,
    token_ids: &[u32],
    device: &<Backend as burn::tensor::backend::Backend>::Device,
) -> Result<()> {
    use voxtral_mini_realtime::tts::pipeline::TtsPipeline;

    let model_dir = PathBuf::from(&args.model);
    if !model_dir.join("consolidated.safetensors").exists() {
        bail!(
            "Model not found at {}\nDownload: hf download mistralai/Voxtral-4B-TTS-2603 --local-dir {}",
            model_dir.join("consolidated.safetensors").display(),
            model_dir.display()
        );
    }

    let start = Instant::now();
    info!("Loading BF16 TTS pipeline from {}", model_dir.display());
    let mut pipeline =
        TtsPipeline::<Backend>::from_model_dir(&model_dir, device).context("Failed to load")?;
    pipeline.set_euler_steps(args.euler_steps);
    info!(
        elapsed_ms = start.elapsed().as_millis() as u64,
        euler_steps = args.euler_steps,
        "BF16 pipeline loaded"
    );

    if args.list_voices {
        let voices = pipeline.list_voices();
        println!("Available voices ({}):", voices.len());
        for name in &voices {
            println!("  {name}");
        }
        return Ok(());
    }

    if !pipeline.has_voice(&args.voice) {
        bail!(
            "Voice '{}' not found. Use --list-voices to see available presets",
            args.voice
        );
    }

    info!(
        text_tokens = token_ids.len(),
        voice = %args.voice,
        "Synthesizing"
    );
    let gen_start = Instant::now();
    let audio =
        pipeline.generate_with_max_frames(token_ids, &args.voice, args.max_frames, device)?;
    let duration = audio.len() as f64 / audio.sample_rate as f64;
    info!(
        elapsed_ms = gen_start.elapsed().as_millis() as u64,
        duration_sec = format!("{duration:.2}"),
        "Audio generated"
    );

    save_audio(&audio, &args.output, gen_start.elapsed(), duration)
}

/// Q4 GGUF path — generic over any CubeCL-backed Burn backend.
fn run_q4<Rt, B>(
    args: &Args,
    gguf_path: &str,
    token_ids: &[u32],
    device: &Rt::Device,
) -> Result<()>
where
    Rt: burn_cubecl::CubeRuntime,
    B: burn::tensor::backend::Backend<
        FloatTensorPrimitive = burn_cubecl::tensor::CubeTensor<Rt>,
        Device = Rt::Device,
    >,
{
    use voxtral_mini_realtime::gguf::Q4TtsModelLoader;
    use voxtral_mini_realtime::tts::config::{AudioCodebookLayout, TtsSpecialTokens};
    use voxtral_mini_realtime::tts::embeddings::AudioCodebookEmbeddings;
    use voxtral_mini_realtime::tts::voice::load_voice_from_bytes;

    let path = PathBuf::from(gguf_path);
    if !path.exists() {
        bail!("GGUF model not found at {}", path.display());
    }

    let start = Instant::now();
    info!("Loading Q4 TTS model from {}", path.display());
    let mut loader = Q4TtsModelLoader::from_file(&path).context("Failed to open GGUF")?;
    // Use load_deferred+finalize to keep token embeddings as Q4 (~216 MB) rather
    // than dequantizing to f32 (~1.5 GB). Works on GPU and CPU/software renderers.
    let (backbone, mut fm, codec) = loader
        .load_deferred::<Rt, B>(device)
        .context("Failed to load Q4 model")?
        .finalize()
        .context("Failed to finalize Q4 model")?;
    fm.set_euler_steps(args.euler_steps);
    info!(
        elapsed_ms = start.elapsed().as_millis() as u64,
        euler_steps = args.euler_steps,
        "Q4 model loaded"
    );

    // Resolve voices directory
    let voices_dir = match &args.voices_dir {
        Some(d) => PathBuf::from(d),
        None => PathBuf::from("models/voxtral-tts/voice_embedding"),
    };

    if args.list_voices {
        if !voices_dir.exists() {
            bail!("Voices directory not found at {}", voices_dir.display());
        }
        let mut voices: Vec<String> = std::fs::read_dir(&voices_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
            .filter_map(|e| {
                e.path()
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
            })
            .collect();
        voices.sort();
        println!("Available voices ({}):", voices.len());
        for name in &voices {
            println!("  {name}");
        }
        return Ok(());
    }

    // Load voice
    let voice_path = voices_dir.join(format!("{}.safetensors", args.voice));
    if !voice_path.exists() {
        bail!(
            "Voice '{}' not found at {}\nUse --list-voices to see available presets",
            args.voice,
            voice_path.display()
        );
    }
    let voice_bytes = std::fs::read(&voice_path)?;
    let voice_embed: Tensor<B, 2> =
        load_voice_from_bytes(&voice_bytes, 3072, device).context("Failed to load voice")?;
    info!(
        voice = %args.voice,
        frames = voice_embed.dims()[0],
        "Voice loaded"
    );

    // Build input sequence
    let special = TtsSpecialTokens::default();
    let bos = backbone.embed_tokens_from_ids(&[special.bos_token_id as i32], 1, 1);
    let begin_audio = backbone.embed_tokens_from_ids(&[special.begin_audio_token_id as i32], 1, 1);
    let next_audio_text =
        backbone.embed_tokens_from_ids(&[special.next_audio_text_token_id as i32], 1, 1);
    let repeat_audio_text =
        backbone.embed_tokens_from_ids(&[special.repeat_audio_text_token_id as i32], 1, 1);
    let text_ids_i32: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
    let text_embeds = backbone.embed_tokens_from_ids(&text_ids_i32, 1, text_ids_i32.len());

    let input_sequence = Tensor::cat(
        vec![
            bos,
            begin_audio.clone(),
            voice_embed.unsqueeze_dim::<3>(0),
            next_audio_text,
            text_embeds,
            repeat_audio_text,
            begin_audio,
        ],
        1,
    );

    let codebook = AudioCodebookEmbeddings::new(
        backbone.audio_codebook_embeddings().clone(),
        AudioCodebookLayout::default(),
    );

    info!(
        text_tokens = token_ids.len(),
        voice = %args.voice,
        "Synthesizing"
    );
    let gen_start = Instant::now();
    let frames = pollster::block_on(backbone.generate_async(
        input_sequence,
        &fm,
        &codebook,
        args.max_frames,
    ))
    .map_err(|e| anyhow::anyhow!("Generation failed: {e}"))?;

    if frames.is_empty() {
        bail!("No audio frames generated");
    }

    // Codec decode
    let n_frames = frames.len();
    let semantic_indices: Vec<usize> = frames.iter().map(|f| f.semantic_idx).collect();
    let mut acoustic_data = Vec::with_capacity(n_frames * 36);
    for frame in &frames {
        for &level in &frame.acoustic_levels {
            acoustic_data.push(level as f32);
        }
    }
    let acoustic_tensor: Tensor<B, 2> = Tensor::from_data(
        burn::tensor::TensorData::new(acoustic_data, [n_frames, 36]),
        device,
    );
    let waveform = codec.decode(&semantic_indices, acoustic_tensor);
    let [_batch, total_samples] = waveform.dims();

    let wav_data = waveform.to_data();
    let mut samples: Vec<f32> = wav_data.as_slice::<f32>().unwrap()[..total_samples].to_vec();

    let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 1e-6 {
        let gain = 0.95 / peak;
        for s in &mut samples {
            *s *= gain;
        }
    }

    let audio = AudioBuffer::new(samples, 24000);
    let duration = audio.len() as f64 / audio.sample_rate as f64;
    info!(
        elapsed_ms = gen_start.elapsed().as_millis() as u64,
        frames = n_frames,
        duration_sec = format!("{duration:.2}"),
        "Audio generated"
    );

    save_audio(&audio, &args.output, gen_start.elapsed(), duration)
}

fn save_audio(
    audio: &AudioBuffer,
    output: &str,
    gen_elapsed: std::time::Duration,
    duration: f64,
) -> Result<()> {
    let output_path = PathBuf::from(output);
    audio
        .save(&output_path)
        .with_context(|| format!("Failed to save {}", output_path.display()))?;

    let rtf = gen_elapsed.as_secs_f64() / duration;
    println!(
        "Saved {:.2}s of audio to {} (RTF {:.2}x)",
        duration,
        output_path.display(),
        rtf,
    );
    Ok(())
}
