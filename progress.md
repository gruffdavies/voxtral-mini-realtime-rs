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

### Phase 2 — Rewrite op.rs kernels with `cube!` macro 🔄 In progress

**Goal:** Replace the `SourceKernel` WGSL dispatch in `src/gguf/op.rs` with `#[cube(launch)]` kernels.  
No type signature changes in this phase. WGPU output must be bit-identical.

**What the old code did:**
- `Q4MatmulNaiveKernel` / `Q4MatmulTiledKernel`: structs implementing `KernelSource`, returning raw WGSL via `include_str!("shader.wgsl")`
- `dispatch()` / `dispatch_naive()`: called `client.launch(Box::new(SourceKernel::new(...)), ...)`

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
- `&Array<T>` for read inputs, `&mut Array<T>` for write outputs (all → `read_write` in WGSL)
- `f32::reinterpret::<u32>(bits)` for bitcast
- `#[comptime]` params passed as Rust values in `::launch(...)` call
- `ArrayArg::from_raw_parts::<T>(&handle, len, 1)` to wrap raw handles

**Compilation errors found and being fixed:**
1. `return` not allowed → use `terminate!()` 
2. `tile_k as u32 < big_k` — `<` parsed as generic args — fix: `(tile_k as u32) < big_k`
3. Secondary "cannot index" errors cascade from #1 and #2

**Current state:** ✅ Kernel compiles, all 231 tests pass, end-to-end TTS is running.

**Verified:**
- `cargo build --features "wgpu,cli,native-tokenizer"` → clean build
- `cargo test --features "wgpu,native-tokenizer"` → 231 unit + 4 integration tests all pass
- End-to-end TTS run: model loads, Q4 backbone runs (our kernel), FM transformer autotuning begins → kernel is working
- Also applied llvmpipe patches to this worktree: `load_deferred`+`finalize` in speak.rs, stub lm_head in tts_loader.rs

**Not yet confirmed:** audio output quality vs. original WGSL kernels (llvmpipe first-run autotuning takes 20+ min). Will compare WAV files once the run completes, or skip and test directly on CUDA in Phase 4.

### Phase 3 — Generics plumbing ⏳ Blocked on Phase 2

Add `<R: CubeRuntime, B: Backend<FloatElem=f32>>` through:
- `src/gguf/tensor.rs` (129 lines, low risk)
- `src/gguf/linear.rs` (117 lines, low risk)
- `src/gguf/model.rs` (~1083 lines, medium)
- `src/gguf/tts_model.rs` (~700 lines, medium)
- `src/gguf/tts_loader.rs` (~861 lines, medium)
- `src/gguf/loader.rs`

### Phase 4 — CUDA end-to-end and benchmark ⏳ Blocked on Phase 3

- Wire `--device cuda` through the full Q4 pipeline
- Run `uv run main.py "Hello world" --device cuda`
- Benchmark RTF vs llvmpipe baseline (~200×)
- Target: real-time or better on RTX 4090

---

## Key files

| File | Status | Notes |
|---|---|---|
| `Cargo.toml` | ✅ Done | `cuda` feature added |
| `src/bin/voxtral/speak.rs` | ✅ Done | `--device` flag, `run_cuda` smoke test |
| `src/gguf/op.rs` | 🔄 In progress | `cube!` kernels replacing WGSL dispatch |
| `src/gguf/shader.wgsl` | Kept as reference | Tiled kernel logic source |
| `src/gguf/shader_naive.wgsl` | Kept as reference | Naive kernel logic source |
| `src/gguf/tensor.rs` | ⏳ Pending | Add `<R: CubeRuntime>` |
| `src/gguf/linear.rs` | ⏳ Pending | Add `<R, B>` |
| `src/gguf/model.rs` | ⏳ Pending | Add `<R, B>` |
| `src/gguf/tts_model.rs` | ⏳ Pending | Add `<R, B>` |
| `src/gguf/tts_loader.rs` | ⏳ Pending | Add `<R, B>` |

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
```
