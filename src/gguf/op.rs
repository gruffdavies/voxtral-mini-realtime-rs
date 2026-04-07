//! Fused Q4_0 dequant+matmul GPU kernel launch.
//!
//! [`q4_matmul`] dispatches a CubeCL compute kernel to perform
//! `output[B, M, N] = input[B, M, K] × weights[N, K]^T` where weights are in
//! Q4_0 format with shape `[N, K]` (out_features, in_features).
//!
//! Two kernel variants are dispatched based on M:
//! - **M ≤ threshold**: Tiled kernel with shared memory.
//!   Cooperatively loads the input vector into workgroup shared memory,
//!   eliminating redundant global reads. Uses 1D (128,1,1) workgroups.
//! - **M > threshold**: Naive kernel.
//!   One thread per output element with (16,16) workgroups — better for
//!   multi-row matmuls where the 2D layout fills the GPU efficiently.
//!
//! On WASM/WebGPU, only the naive kernel is used.
//!
//! Kernels are written using the CubeCL `#[cube(launch)]` macro, which
//! compiles the same source to WGSL (for wgpu) and PTX (for CUDA), making
//! this the portability layer for the Q4 path.

use burn::tensor::backend::Backend;
use burn::tensor::{DType, Tensor, TensorPrimitive};
use burn_cubecl::{CubeRuntime, tensor::CubeTensor};
use cubecl::client::ComputeClient;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::tensor::Q4Tensor;

/// M threshold: use tiled kernel when M ≤ this, naive kernel otherwise.
#[cfg(not(target_family = "wasm"))]
const TILED_M_THRESHOLD: usize = 4;

/// Workgroup size X for the tiled kernel (1D workgroups).
/// 32 threads per block (1 warp) gives ceil(N/32) blocks. For N=3072 that is
/// 96 blocks across 128 SMs — ~75% utilisation vs 9% with 256.
const TILED_WG_X: usize = 32;

/// Tile size for K-dimension shared memory (must be a multiple of 32).
/// 256 keeps each thread's per-tile load at 8 F32s (32 bytes) — coalesced
/// within a single warp, low shared-memory pressure (1 KB per block).
const TILE_K: usize = 256;

/// Workgroup size X for the naive kernel (2D workgroups).
const NAIVE_WG_X: usize = 16;

/// Workgroup size Y for the naive kernel (2D workgroups).
const NAIVE_WG_Y: usize = 16;

// ---------------------------------------------------------------------------
// CubeCL kernel helper functions
// ---------------------------------------------------------------------------

/// Convert the lower 16 bits of a u32 (an f16 bit pattern) to an f32.
///
/// Uses manual bit manipulation to reconstruct the f32 representation:
///   f32_exp = f16_exp + 112  (bias difference: 127 − 15 = 112)
///   f32_frac = f16_frac shifted left by 13 bits
///
/// This handles all normal f16 values, which is the only kind that appears
/// as a Q4_0 block scale in practice.
#[cube]
fn f16_bits_to_f32(bits: u32) -> f32 {
    let sign = (bits >> 15u32) & 1u32;
    let exp = (bits >> 10u32) & 0x1Fu32;
    let frac = bits & 0x3FFu32;
    // Rebuild an f32 bit pattern from the f16 components.
    // For exp == 0 (subnormal f16) this overflows into a normal f32, but
    // Q4_0 scales are always normal values so this is acceptable.
    let f32_bits = (sign << 31u32) | ((exp + 112u32) << 23u32) | (frac << 13u32);
    f32::reinterpret::<u32>(f32_bits)
}

// ---------------------------------------------------------------------------
// Naive kernel: one thread per output element
// ---------------------------------------------------------------------------
//
// Grid: ceil(N / NAIVE_WG_X) × ceil(B*M / NAIVE_WG_Y) cubes,
//       each cube is (NAIVE_WG_X, NAIVE_WG_Y, 1).
//
// Thread (gid.x, gid.y) computes output[b, m, n] where:
//   n   = gid.x
//   bm  = gid.y  (= b * M + m)

#[cube(launch)]
fn q4_matmul_naive(
    weights: &mut Array<u32>,
    input: &mut Array<f32>,
    output: &mut Array<f32>,
    info: &mut Array<u32>,
) {
    let n = ABSOLUTE_POS_X;
    let bm = ABSOLUTE_POS_Y;

    let big_b = info[0usize];
    let big_m = info[1usize];
    let big_k = info[2usize];
    let big_n = info[3usize];
    let blocks_per_row = info[4usize];

    let row_m = bm % big_m;
    let b = bm / big_m;

    if n >= big_n || b >= big_b {
        terminate!();
    }

    let mut acc = 0.0f32;
    let input_base = b * big_m * big_k + row_m * big_k;

    for blk in 0u32..blocks_per_row {
        // Aligned 20-byte block layout: 5 u32s per block.
        //   u32[0]: F16 scale in bits [15:0]
        //   u32[1..4]: 4 × u32 of packed Q4 nibbles
        let block_u32 = (n * blocks_per_row + blk) * 5u32;
        let scale_bits = weights[block_u32 as usize] & 0xFFFFu32;
        let scale = f16_bits_to_f32(scale_bits);
        let k_base = blk * 32u32;

        for wi in 0u32..4u32 {
            let packed = weights[(block_u32 + 1u32 + wi) as usize];
            let b0 = packed & 0xFFu32;
            let b1 = (packed >> 8u32) & 0xFFu32;
            let b2 = (packed >> 16u32) & 0xFFu32;
            let b3 = (packed >> 24u32) & 0xFFu32;

            let base_i = wi * 4u32;
            let k_off = (input_base + k_base + base_i) as usize;

            // Lower nibbles → elements [base_i .. base_i+3]
            acc += ((b0 & 0xFu32) as f32 - 8.0f32) * scale * input[k_off];
            acc += ((b1 & 0xFu32) as f32 - 8.0f32) * scale * input[k_off + 1usize];
            acc += ((b2 & 0xFu32) as f32 - 8.0f32) * scale * input[k_off + 2usize];
            acc += ((b3 & 0xFu32) as f32 - 8.0f32) * scale * input[k_off + 3usize];

            // Upper nibbles → elements [16 + base_i .. 16 + base_i+3]
            let k_off_hi = k_off + 16usize;
            acc += (((b0 >> 4u32) & 0xFu32) as f32 - 8.0f32) * scale * input[k_off_hi];
            acc += (((b1 >> 4u32) & 0xFu32) as f32 - 8.0f32) * scale * input[k_off_hi + 1usize];
            acc += (((b2 >> 4u32) & 0xFu32) as f32 - 8.0f32) * scale * input[k_off_hi + 2usize];
            acc += (((b3 >> 4u32) & 0xFu32) as f32 - 8.0f32) * scale * input[k_off_hi + 3usize];
        }
    }

    output[(b * big_m * big_n + row_m * big_n + n) as usize] = acc;
}

// ---------------------------------------------------------------------------
// Tiled kernel: 128-thread workgroups cooperatively load input into shared memory
// ---------------------------------------------------------------------------
//
// Grid: ceil(N / TILED_WG_X) × (B*M) cubes, each cube is (TILED_WG_X, 1, 1).
//
// All TILED_WG_X threads in a cube cooperatively load a TILE_K-sized slice
// of the input row into shared memory before each accumulates against its own
// weight row. This eliminates redundant global memory reads for M=1
// (autoregressive decode). Weight reads use the aligned 5-u32-per-block
// format — no byte-offset arithmetic.
//
// The `valid` flag ensures all threads reach sync_cube() even when b ≥ B
// (out-of-bounds cube in the Y direction is possible when B*M is not a
// multiple of the Y grid stride — in the 1D Y case it never is, but left
// here for safety parity with the WGSL original).

#[cfg(not(target_family = "wasm"))]
#[cube(launch)]
fn q4_matmul_tiled(
    weights: &mut Array<u32>,
    input: &mut Array<f32>,
    output: &mut Array<f32>,
    info: &mut Array<u32>,
    #[comptime] tile_k: usize,
    #[comptime] wg_size: usize,
) {
    let mut shared_input = SharedMemory::<f32>::new(tile_k);

    let n = ABSOLUTE_POS_X;
    let bm = ABSOLUTE_POS_Y;

    let big_b = info[0usize];
    let big_m = info[1usize];
    let big_k = info[2usize];
    let big_n = info[3usize];
    let blocks_per_row = info[4usize];

    let row_m = bm % big_m;
    let b = bm / big_m;

    // `valid` prevents early returns that would skip sync_cube() calls.
    let valid = b < big_b;
    let mut acc = 0.0f32;
    // When !valid (b >= B), input_base may be out of bounds but is never used
    // because the cooperative load and accumulate loops are guarded by `valid`.
    let input_base = b * big_m * big_k + row_m * big_k;

    let num_tiles = (big_k + tile_k as u32 - 1u32) / tile_k as u32;

    for tile in 0u32..num_tiles {
        let tile_start = tile * tile_k as u32;

        // Cooperative load: all threads fill their slice of shared_input.
        let mut k_local = UNIT_POS_X as usize;
        while k_local < tile_k {
            let k_global = tile_start as usize + k_local;
            if valid && k_global < big_k as usize {
                shared_input[k_local] = input[input_base as usize + k_global];
            }
            k_local += wg_size;
        }
        sync_cube();

        // Accumulate against shared input.
        if valid && n < big_n {
            let tile_end = if tile_start + (tile_k as u32) < big_k {
                tile_start + (tile_k as u32)
            } else {
                big_k
            };
            let blocks_in_tile = (tile_end - tile_start) / 32u32;
            let block_base = tile_start / 32u32;

            for blk in 0u32..blocks_in_tile {
                // Aligned 20-byte block layout: 5 u32s per block.
                //   u32[0]: F16 scale in bits [15:0]
                //   u32[1..4]: 4 × u32 of packed Q4 nibbles
                let global_block = n * blocks_per_row + block_base + blk;
                let block_u32 = global_block * 5u32;
                let scale_bits = weights[block_u32 as usize] & 0xFFFFu32;
                let scale = f16_bits_to_f32(scale_bits);
                let k_base = (blk * 32u32) as usize;

                for wi in 0u32..4u32 {
                    let packed = weights[(block_u32 + 1u32 + wi) as usize];
                    let b0 = packed & 0xFFu32;
                    let b1 = (packed >> 8u32) & 0xFFu32;
                    let b2 = (packed >> 16u32) & 0xFFu32;
                    let b3 = (packed >> 24u32) & 0xFFu32;

                    let base_i = (wi * 4u32) as usize;
                    let sm_off = k_base + base_i;

                    // Lower nibbles → shared_input[k_base + base_i .. +3]
                    acc += ((b0 & 0xFu32) as f32 - 8.0f32) * scale * shared_input[sm_off];
                    acc += ((b1 & 0xFu32) as f32 - 8.0f32) * scale * shared_input[sm_off + 1usize];
                    acc += ((b2 & 0xFu32) as f32 - 8.0f32) * scale * shared_input[sm_off + 2usize];
                    acc += ((b3 & 0xFu32) as f32 - 8.0f32) * scale * shared_input[sm_off + 3usize];

                    // Upper nibbles → shared_input[k_base + 16 + base_i .. +3]
                    let sm_off_hi = sm_off + 16usize;
                    acc += (((b0 >> 4u32) & 0xFu32) as f32 - 8.0f32) * scale * shared_input[sm_off_hi];
                    acc += (((b1 >> 4u32) & 0xFu32) as f32 - 8.0f32) * scale * shared_input[sm_off_hi + 1usize];
                    acc += (((b2 >> 4u32) & 0xFu32) as f32 - 8.0f32) * scale * shared_input[sm_off_hi + 2usize];
                    acc += (((b3 >> 4u32) & 0xFu32) as f32 - 8.0f32) * scale * shared_input[sm_off_hi + 3usize];
                }
            }
        }
        sync_cube();
    }

    if n < big_n && b < big_b {
        output[(b * big_m * big_n + row_m * big_n + n) as usize] = acc;
    }
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

/// Fused Q4_0 dequant+matmul on GPU.
///
/// Computes `output[B, M, N] = input[B, M, K] × weights[N, K]^T` where
/// weights are stored in Q4_0 block format on the GPU with shape `[N, K]`
/// (out_features, in_features). Dequantization happens inside the kernel —
/// no intermediate full-precision weight buffer is created.
///
/// `R` is the CubeCL runtime (e.g. `WgpuRuntime` or `CudaRuntime`);
/// `B` is the Burn backend whose float primitive is `CubeTensor<R>`,
/// i.e. some `CubeBackend<R, f32, ...>`.
pub fn q4_matmul<R, B>(input: Tensor<B, 3>, weights: &Q4Tensor<R>) -> Tensor<B, 3>
where
    R: CubeRuntime,
    B: Backend<FloatTensorPrimitive = CubeTensor<R>>,
{
    use burn_cubecl::kernel::into_contiguous;
    use burn::tensor::TensorPrimitive;

    let cube_input: CubeTensor<R> = input.into_primitive().tensor();
    let cube_input = into_contiguous(cube_input);

    assert_eq!(cube_input.shape.num_dims(), 3, "Input must be 3D [B, M, K]");
    let b = cube_input.shape.dims[0];
    let m = cube_input.shape.dims[1];
    let k = cube_input.shape.dims[2];
    let [n, wk] = weights.shape();
    assert_eq!(
        k, wk,
        "K dimension mismatch: input has {k}, weights have {wk}"
    );

    let client = cube_input.client.clone();
    let device = cube_input.device.clone();
    let blocks_per_row = k / 32;

    let output_handle = client.empty(b * m * n * 4);

    let info: [u32; 5] = [
        b as u32,
        m as u32,
        k as u32,
        n as u32,
        blocks_per_row as u32,
    ];
    let info_bytes: Vec<u8> = info.iter().flat_map(|v| v.to_le_bytes()).collect();
    let info_handle = client.create_from_slice(&info_bytes);

    // Weights buffer: 5 u32s per block (aligned 20-byte format).
    let weights_u32_len = weights.num_blocks() * 5;

    #[cfg(feature = "cuda")]
    let _nvtx = crate::nvtx::Guard::new(&format!("q4mm {m}x{n}"));

    dispatch(
        &client,
        b,
        m,
        n,
        &weights.handle,
        weights_u32_len,
        &cube_input.handle,
        b * m * k,
        &output_handle,
        b * m * n,
        &info_handle,
    );

    let output_tensor = CubeTensor::new_contiguous(
        client,
        device,
        burn::prelude::Shape::from(vec![b, m, n]),
        output_handle,
        DType::F32,
    );
    Tensor::from_primitive(TensorPrimitive::Float(output_tensor))
}

/// Dispatch the appropriate kernel variant.
///
/// On native: tiled for M ≤ TILED_M_THRESHOLD, naive otherwise.
/// On WASM: always naive.
#[cfg(not(target_family = "wasm"))]
#[allow(clippy::too_many_arguments)]
fn dispatch<R: CubeRuntime>(
    client: &ComputeClient<R>,
    b: usize,
    m: usize,
    n: usize,
    weights_handle: &Handle,
    weights_u32_len: usize,
    input_handle: &Handle,
    input_len: usize,
    output_handle: &Handle,
    output_len: usize,
    info_handle: &Handle,
) {
    if m <= TILED_M_THRESHOLD {
        let wg_x = n.div_ceil(TILED_WG_X) as u32;
        let wg_y = (b * m) as u32;
        unsafe {
            q4_matmul_tiled::launch(
                client,
                CubeCount::new_2d(wg_x, wg_y),
                CubeDim::new_1d(TILED_WG_X as u32),
                ArrayArg::from_raw_parts::<u32>(weights_handle, weights_u32_len, 1),
                ArrayArg::from_raw_parts::<f32>(input_handle, input_len, 1),
                ArrayArg::from_raw_parts::<f32>(output_handle, output_len, 1),
                ArrayArg::from_raw_parts::<u32>(info_handle, 5, 1),
                TILE_K,
                TILED_WG_X,
            )
            .expect("Q4 tiled matmul kernel launch failed");
        }
    } else {
        dispatch_naive(
            client,
            b,
            m,
            n,
            weights_handle,
            weights_u32_len,
            input_handle,
            input_len,
            output_handle,
            output_len,
            info_handle,
        );
    }
}

#[cfg(target_family = "wasm")]
#[allow(clippy::too_many_arguments)]
fn dispatch<R: CubeRuntime>(
    client: &ComputeClient<R>,
    b: usize,
    m: usize,
    n: usize,
    weights_handle: &Handle,
    weights_u32_len: usize,
    input_handle: &Handle,
    input_len: usize,
    output_handle: &Handle,
    output_len: usize,
    info_handle: &Handle,
) {
    dispatch_naive(
        client,
        b,
        m,
        n,
        weights_handle,
        weights_u32_len,
        input_handle,
        input_len,
        output_handle,
        output_len,
        info_handle,
    );
}

#[allow(clippy::too_many_arguments)]
fn dispatch_naive<R: CubeRuntime>(
    client: &ComputeClient<R>,
    b: usize,
    m: usize,
    n: usize,
    weights_handle: &Handle,
    weights_u32_len: usize,
    input_handle: &Handle,
    input_len: usize,
    output_handle: &Handle,
    output_len: usize,
    info_handle: &Handle,
) {
    let wg_x = n.div_ceil(NAIVE_WG_X) as u32;
    let wg_y = (b * m).div_ceil(NAIVE_WG_Y) as u32;
    unsafe {
        q4_matmul_naive::launch(
            client,
            CubeCount::new_2d(wg_x, wg_y),
            CubeDim::new_2d(NAIVE_WG_X as u32, NAIVE_WG_Y as u32),
            ArrayArg::from_raw_parts::<u32>(weights_handle, weights_u32_len, 1),
            ArrayArg::from_raw_parts::<f32>(input_handle, input_len, 1),
            ArrayArg::from_raw_parts::<f32>(output_handle, output_len, 1),
            ArrayArg::from_raw_parts::<u32>(info_handle, 5, 1),
        )
        .expect("Q4 naive matmul kernel launch failed");
    }
}
