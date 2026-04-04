# CUDA Backend Refactor — Progress Log

Worktree: `voxtral-mini-realtime-rs-cuda/`  
Branch: `cuda-backend`  
Goal: Replace hardcoded `Wgpu` throughout `src/gguf/` with `<R: CubeRuntime, B: Backend>` generics so inference can run on CUDA (RTX 4090) instead of llvmpipe (~200× RTF).

---

## Phases

### Phase 1 — CUDA smoke test ✅ Complete

- Added `cuda = ["burn/cuda", "cubecl/cuda"]` feature to `Cargo.toml`
- Added `burn-cuda = { version = "0.20", optional = true }` dependency
- Added `DeviceArg` enum and `--device` CLI flag to `src/bin/voxtral/speak.rs`
- `run_cuda()` allocates `Tensor::<Cuda, 1>::zeros([4], &device)` and confirms the RTX 4090 is active
- Build: `cargo build --bin voxtral --features "wgpu,cli,native-tokenizer,cuda"`
- Test: `./target/release/voxtral speak --device cuda --text "hi"` → prints CUDA device info, bail with "smoke test passed"

### Phase 2 — Rewrite op.rs kernels with `cube!` macro ✅ Complete

**Goal:** Replace the `SourceKernel` WGSL dispatch in `src/gguf/op.rs` with `#[cube(launch)]` kernels.  
No type signature changes in this phase. WGPU output must be bit-identical.

**What the new code does:**
- `read_u32_unaligned` helper: `#[cube]` fn, reads u32 at byte-unaligned offset (matches WGSL helper)
- `f16_bits_to_f32` helper: `#[cube]` fn, manual f16→f32 bit manipulation (replaces `unpack2x16float`)
- `q4_matmul_naive`: `#[cube(launch)]` fn, 1 thread per output, (16×16) workgroups
- `q4_matmul_tiled`: `#[cube(launch)]` fn, cooperative shared-memory load, 128×1 workgroups
- Dispatch functions call `q4_matmul_naive::launch(...)` / `q4_matmul_tiled::launch(...)` with `ArrayArg::from_raw_parts`

**CubeCL API notes learned:**
- `SharedMemory::<f32>::new(#[comptime] size)` for workgroup shared memory
- `sync_cube()` = workgroup barrier
- `ABSOLUTE_POS_X`, `ABSOLUTE_POS_Y` are `u32`; `UNIT_POS_X` is `u32`
- `terminate!()` instead of `return` (early exit in cube kernels)
- `&mut Array<T>` for ALL array params (even read-only; all GPU storage buffers are `read_write`)
- `f32::reinterpret::<u32>(bits)` for bitcast
- `#[comptime]` params passed as Rust values in `::launch(...)` call
- `ArrayArg::from_raw_parts::<T>(&handle, len, 1)` to wrap raw handles

**Verified:**
- `cargo build --features "wgpu,cli,native-tokenizer"` → clean build
- `cargo test --features "wgpu,native-tokenizer"` → 231 unit + 4 integration tests pass
- End-to-end TTS run: model loads, Q4 backbone runs (our kernel), FM transformer autotuning begins

### Phase 3 — Generics plumbing ✅ Complete

**Goal:** Add `<R: CubeRuntime, B: Backend<FloatTensorPrimitive = CubeTensor<R>, Device = R::Device>>`
through all `src/gguf/` files so the pipeline compiles for any CubeCL-backed Burn backend.

**Key design decisions:**
- Used `B: Backend<FloatTensorPrimitive = CubeTensor<R>, Device = R::Device>` to express that the device types must match (true for all CubeCL backends: `Wgpu::Device = WgpuRuntime::Device = WgpuDevice`, etc.)
- Added `PhantomData<B>` marker to `Q4FusedQKV` and `Q4FusedGateUp` (no tensor field uses `B` directly)
- I/O reader type renamed `Rdr: Read + Seek` in loaders (avoids collision with `R: CubeRuntime`)
- Loader methods take separate `<Rt, B>` type params: `loader.load::<WgpuRuntime, Wgpu>(&device)`
- `fuse_qkv` / `fuse_gate_up` device param changed from `&WgpuDevice` to `&R::Device`
- `Q4LanguageModel::device` field changed from `WgpuDevice` to `R::Device`

**Files changed:**
- `src/gguf/tensor.rs` — `Q4Tensor<R: CubeRuntime>` (done in earlier context)
- `src/gguf/linear.rs` — `Q4Linear<R, B>`, `Q4FusedQKV<R, B>`, `Q4FusedGateUp<R, B>`
- `src/gguf/model.rs` — all structs and impls, ~1083 lines rewritten
- `src/gguf/tts_model.rs` — all structs and impls, ~700 lines rewritten
- `src/gguf/loader.rs` — `Q4ModelParts<Rt,B>`, all load methods generic
- `src/gguf/tts_loader.rs` — `Q4TtsModelParts<Rt,B>`, all load methods generic
- `src/bin/voxtral/transcribe.rs` — `Q4VoxtralModel<WgpuRuntime, Backend>` concrete types
- `src/bin/voxtral/speak.rs` — `load_deferred::<WgpuRuntime, Wgpu>` explicit type args
- `src/gguf/tests.rs` — `TestRuntime = WgpuRuntime`, explicit closure return types

**Verified:**
- `cargo build --features "wgpu,cli,native-tokenizer"` → clean (4 minor warnings)
- `cargo build --features "wgpu,cli,native-tokenizer,cuda"` → clean (CUDA path compiles too)
- `cargo test --features "wgpu,native-tokenizer"` → **239 tests pass** (231 unit + 4 integration + 3 tts_model + 1 tts integration)

### Phase 6 — SM utilisation fix: TILED_WG_X 256→32, TILE_K 1024→256 ✅ Complete

**Root cause identified:** With `TILED_WG_X=256`, grid size = `ceil(N/256)` blocks.
For the dominant backbone shapes (N=3072, 4096) that launches 12–16 blocks on a
GPU with 128 SMs — 9–12% SM utilisation. The GPU was mostly idle.

**Fix:** `TILED_WG_X=32` (1 warp per block) → `ceil(N/32)` blocks:
- N=3072 → 96 blocks (75% utilisation)
- N=4096 → 128 blocks (100%)
- N=9216 → 288 blocks (225%, 2+ blocks/SM for latency hiding)

`TILE_K=256` keeps per-thread tile load at 8 F32s (32 bytes) — one coalesced
warp-width transaction. Shared memory drops from 4 KB to 1 KB per block.

**Benchmark (RTX 4090, warm cache, CUBECL_AUTOTUNE_LEVEL=minimal):**

| Text | Before (WGX=256) | After (WGX=32) | Δ |
|---|---|---|---|
| "Hello world" | 21.5× | **15.2×** | −29% |
| Long sentence | 8.4× | **6.4×** | −24% |

**Why not the theoretical 3–8×?** Q4 matmul speedup diluted by:
- Burn's standard CUDA matmul for attention scores
- RMS norm, RoPE, softmax (element-wise Burn ops)
- Kernel-launch overhead (~100+ dispatches per decode step)

Overall improvement from original baseline (RTF 30×): **2× total**.
Not real-time. Remaining bottlenecks are outside the Q4 kernel.

### Phase 5 — Option 1: aligned weights + kernel constant tuning ✅ Complete

**Goal:** Eliminate `read_u32_unaligned` overhead and tune kernel constants for 4090.

**Changes:**
- `tensor.rs`: `from_q4_bytes` now repacks 18-byte Q4_0 GGUF blocks to 20-byte aligned format
  (5 u32s: `[scale_f16|0x0000, data_u32×4]`). `read_bytes` unpacks back to 18-byte for
  API compatibility. `dequantize` updated to read new layout.
- `op.rs`: Removed `read_u32_unaligned`. Kernels now use `block_u32 = global_block * 5u32` with
  direct indexed u32 reads. `TILED_WG_X` 128 → 256. `TILE_K` 512 → 1024.

**Benchmark (RTX 4090, warm autotune cache, CUBECL_AUTOTUNE_LEVEL=minimal):**

| Text | Duration | RTF | vs. Phase 4 baseline |
|---|---|---|---|
| "Hello world" | 1.60s | 21.5× | 30.3× → 21.5× (−29%) |
| Long sentence (~17 tokens) | 6.80s | 8.4× | ~9.3× → 8.4× (−10%) |

**CubeCL 0.9.0 bug (unrelated to our changes):**
`simple_async_mma` (a conv forward kernel) is selected by autotune for certain codec
decoder shapes but then fails during actual launch with:
> "Too many data will be loaded … total unit count 128 divides number of lines in stage"
Workaround: `CUBECL_AUTOTUNE_LEVEL=minimal` uses coarser key anchoring (scale factor 1.25)
which bins the problematic shapes into different buckets that don't select the async MMA variant.
Set via `env.setdefault("CUBECL_AUTOTUNE_LEVEL", "minimal")` in `main.py` for CUDA runs.

### Phase 4 — CUDA end-to-end and benchmark ✅ Complete

**Goal:** Replace the `run_cuda()` smoke test bail with actual Q4 inference, benchmark vs llvmpipe.

**What was done:**
- `run_cuda()` already wired to `run_q4::<CudaRuntime, CudaBackend>` from Phase 3 session (discovered on resume)
- Backend type: `CubeBackend<CudaRuntime, f32, i32, u8>` (non-fusion, satisfies `FloatTensorPrimitive = CubeTensor<R>` constraint)
- Added `--device cuda` to Python wrapper `main.py`; auto-selects cuda worktree binary

**Benchmark (RTX 4090, WSL2):**

| Run | Text | Duration | RTF |
|---|---|---|---|
| Cold (first run, autotuning) | "Hello world" (2 tokens) | 1.2s audio / 79s wall | 65.6× |
| Warm | "Hello world, this is a test…" (longer) | 7.8s audio / 73s wall | 9.3× |
| Warm | "Hello world" (2 tokens) | 2.5s audio / 76s wall | 30.3× |

RTF ~9–30× on warm cache (depends on sequence length / amortisation). llvmpipe baseline was ~200×.
Autotuning is one-time per problem shape; cached in `~/.cache/burn/...` for subsequent runs.

**Note:** Not real-time yet. The Q4 matmul kernel (CubeCL-generated PTX) is not fused with dequantization and hits many small matmul shapes (batch=1, token-by-token decoding). cuBLAS-backed fusion would help significantly.

---

## Key files

| File | Status | Notes |
|---|---|---|
| `Cargo.toml` | ✅ Done | `cuda` feature added |
| `src/bin/voxtral/speak.rs` | 🔄 Phase 4 | WGPU path done; CUDA path still smoke-test bail |
| `src/gguf/op.rs` | ✅ Done | `cube!` kernels, generic dispatch functions |
| `src/gguf/tensor.rs` | ✅ Done | `Q4Tensor<R: CubeRuntime>` |
| `src/gguf/linear.rs` | ✅ Done | `Q4Linear<R, B>` etc. |
| `src/gguf/model.rs` | ✅ Done | All model structs generic |
| `src/gguf/tts_model.rs` | ✅ Done | All TTS model structs generic |
| `src/gguf/tts_loader.rs` | ✅ Done | `Q4TtsModelParts<Rt,B>`, generic loaders |
| `src/gguf/loader.rs` | ✅ Done | `Q4ModelParts<Rt,B>`, generic loaders |

---

## Build commands

```bash
# WGPU (default)
cd /home/gruff/projects/Sandpit/Audio/voxtral-mini-realtime-rs-cuda
cargo build --bin voxtral --features "wgpu,cli,native-tokenizer"

# CUDA
cargo build --bin voxtral --features "wgpu,cli,native-tokenizer,cuda"

# Run tests
cargo test --features "wgpu,native-tokenizer"

# TTS inference (WGPU)
uv run main.py "Hello world"
```

---

## Appendix: CubeCL 0.9 Kernel API Quick Reference

| CubeCL | Equivalent |
|---|---|
| `terminate!()` | `return` in kernel |
| `sync_cube()` | workgroup barrier |
| `ABSOLUTE_POS_X` | global thread X (u32) |
| `UNIT_POS_X` | local thread X (u32) |
| `SharedMemory::<f32>::new(#[comptime] N)` | workgroup shared memory |
| `f32::reinterpret::<u32>(bits)` | bitcast f32 → u32 |
| `&mut Array<T>` | all GPU array params (always read_write) |
| `ArrayArg::from_raw_parts::<T>(&handle, len, 1)` | wrap raw handle for launch |
| `#[comptime] n: u32` | compile-time constant param |
