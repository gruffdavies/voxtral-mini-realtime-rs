//! Q4_0 quantized weight tensor stored on GPU.
//!
//! [`Q4Tensor`] is generic over `R: CubeRuntime` so the same struct works for
//! both the WGPU (default) and CUDA paths. It uploads raw Q4_0 blocks to a
//! GPU storage buffer and exposes them for the fused dequant+matmul kernel in
//! [`super::op`].

use anyhow::{ensure, Result};
use burn_cubecl::CubeRuntime;
use cubecl::client::ComputeClient;
use cubecl::server::Handle;

/// A Q4_0 quantized weight tensor living on GPU.
///
/// Weights are stored in an aligned 20-byte-per-block format (5 u32s):
///   u32[0]: F16 scale in bits [15:0], bits [31:16] zero-padded
///   u32[1..4]: 16 bytes of packed Q4 nibbles
/// This differs from the 18-byte GGUF on-disk format; repacking happens in
/// [`Q4Tensor::from_q4_bytes`] so GPU kernels can use direct indexed u32
/// reads with no byte-offset arithmetic.
pub struct Q4Tensor<R: CubeRuntime> {
    pub(crate) handle: Handle,
    shape: [usize; 2],
    num_blocks: usize,
    client: ComputeClient<R>,
    device: R::Device,
}

impl<R: CubeRuntime> Q4Tensor<R> {
    /// Upload raw Q4_0 bytes to a GPU storage buffer, repacking to aligned format.
    ///
    /// Shape is `[N, K]` = `[out_features, in_features]`, matching PyTorch/GGUF
    /// convention. `raw_bytes` must contain exactly `(N * K / 32) * 18` bytes
    /// in GGUF Q4_0 format (18 bytes per block).
    ///
    /// On upload the blocks are repacked to 20-byte aligned format (5 u32s per
    /// block) so GPU kernels can use direct indexed u32 reads with no
    /// byte-offset arithmetic. See struct docstring for layout details.
    pub fn from_q4_bytes(raw_bytes: &[u8], shape: [usize; 2], device: &R::Device) -> Result<Self> {
        let [n, k] = shape;
        let num_elements = k * n;
        ensure!(
            num_elements % 32 == 0,
            "Q4_0 requires element count divisible by 32, got {num_elements}"
        );
        let num_blocks = num_elements / 32;
        let expected_bytes = num_blocks * 18;
        ensure!(
            raw_bytes.len() == expected_bytes,
            "Q4_0 byte count mismatch: expected {expected_bytes} for {num_blocks} blocks, got {}",
            raw_bytes.len()
        );

        let client = R::client(device);

        // Repack from GGUF 18-byte blocks to aligned 20-byte blocks (5 u32s):
        //   bytes [0..2)  → F16 scale (same position)
        //   bytes [2..4)  → 0x0000 padding
        //   bytes [4..20) → 16 bytes Q4 data (was at [2..18))
        let mut repacked = Vec::with_capacity(num_blocks * 20);
        for i in 0..num_blocks {
            let src = &raw_bytes[i * 18..(i + 1) * 18];
            repacked.push(src[0]);  // F16 scale lo
            repacked.push(src[1]);  // F16 scale hi
            repacked.push(0u8);     // padding
            repacked.push(0u8);     // padding
            repacked.extend_from_slice(&src[2..18]); // 16 bytes Q4 nibbles
        }
        let handle = client.create_from_slice(&repacked);

        Ok(Self {
            handle,
            shape,
            num_blocks,
            client,
            device: device.clone(),
        })
    }

    /// Logical weight dimensions `[N, K]` = `[out_features, in_features]`.
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    /// Number of Q4_0 blocks in the tensor.
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Read raw Q4_0 bytes from GPU, unpacking back to GGUF 18-byte format.
    pub fn read_bytes(&self) -> Vec<u8> {
        let raw = self.client.read_one(self.handle.clone());
        let expected_aligned = self.num_blocks * 20;
        assert!(
            raw.len() >= expected_aligned,
            "Q4Tensor::read_bytes: GPU buffer shorter than expected: got {}, expected {expected_aligned}",
            raw.len()
        );
        // Unpack aligned 20-byte format → GGUF 18-byte format
        let mut out = Vec::with_capacity(self.num_blocks * 18);
        for i in 0..self.num_blocks {
            let base = i * 20;
            out.extend_from_slice(&raw[base..base + 2]);      // F16 scale
            out.extend_from_slice(&raw[base + 4..base + 20]); // 16 bytes Q4 data
        }
        out
    }

    /// Compute client for this tensor.
    pub(crate) fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    /// Device this tensor lives on.
    pub(crate) fn device(&self) -> &R::Device {
        &self.device
    }
}

// ---------------------------------------------------------------------------
// WGPU-only methods
// ---------------------------------------------------------------------------

#[cfg(not(target_family = "wasm"))]
impl Q4Tensor<burn::backend::wgpu::WgpuRuntime> {
    /// Dequantize the Q4_0 data to a full-precision `Tensor<Wgpu, 2>`.
    ///
    /// Reads raw bytes from GPU and dequantizes on CPU.
    /// Intended for diagnostics and testing — the hot path uses
    /// [`q4_matmul`](super::op::q4_matmul) which dequantizes on GPU.
    pub fn dequantize(&self) -> burn::tensor::Tensor<burn::backend::Wgpu, 2> {
        use burn::tensor::{Tensor, TensorData};

        let bytes = self.client.read_one(self.handle.clone());
        let raw: &[u8] = &bytes;

        let [n, k] = self.shape;
        let num_elements = n * k;
        let mut output = vec![0.0f32; num_elements];

        for block_idx in 0..self.num_blocks {
            let offset = block_idx * 20; // 20-byte aligned blocks on GPU
            let d_bits = u16::from_le_bytes([raw[offset], raw[offset + 1]]);
            let d = half::f16::from_bits(d_bits).to_f32();

            let base = block_idx * 32;
            for i in 0..16 {
                let byte = raw[offset + 4 + i]; // data starts at byte 4 (after scale + 2-byte pad)
                let lo = (byte & 0x0F) as f32 - 8.0;
                let hi = ((byte >> 4) & 0x0F) as f32 - 8.0;
                output[base + i] = lo * d;
                output[base + i + 16] = hi * d;
            }
        }

        let tensor_data = TensorData::new(output, [n, k]);
        Tensor::from_data(tensor_data, &self.device)
    }
}

/// WASM path: dequantize not available (no direct GPU readback in browsers).
#[cfg(target_family = "wasm")]
impl Q4Tensor<burn::backend::wgpu::WgpuRuntime> {
    pub fn dequantize(&self) -> burn::tensor::Tensor<burn::backend::Wgpu, 2> {
        panic!("Q4Tensor::dequantize not available on WASM")
    }
}
