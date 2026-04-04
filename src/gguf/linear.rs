//! Q4_0 quantized linear layer.
//!
//! [`Q4Linear`] wraps a [`Q4Tensor`] weight matrix and optional f32 bias,
//! providing a `forward` method that delegates to [`q4_matmul`].

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::CubeRuntime;

use super::op::q4_matmul;
use super::tensor::Q4Tensor;

/// A linear layer with Q4_0 quantized weights.
///
/// Stores weights as `[out_features, in_features]` in Q4_0 format and an
/// optional f32 bias vector. The forward pass computes
/// `x @ weights^T + bias` via the fused dequant+matmul GPU kernel.
pub struct Q4Linear<R, B>
where
    R: CubeRuntime,
    B: Backend<FloatTensorPrimitive = CubeTensor<R>, Device = R::Device>,
{
    weights: Q4Tensor<R>,
    bias: Option<Tensor<B, 1>>,
}

impl<R, B> Q4Linear<R, B>
where
    R: CubeRuntime,
    B: Backend<FloatTensorPrimitive = CubeTensor<R>, Device = R::Device>,
{
    /// Create a new Q4 linear layer.
    ///
    /// `weights` shape must be `[out_features, in_features]`.
    pub fn new(weights: Q4Tensor<R>, bias: Option<Tensor<B, 1>>) -> Self {
        Self { weights, bias }
    }

    /// Access the underlying Q4 weight tensor.
    pub fn weights(&self) -> &Q4Tensor<R> {
        &self.weights
    }

    /// Forward pass: `x @ weights^T + bias`.
    ///
    /// `x` shape: `[B, M, K]` where `K = in_features`.
    /// Returns shape: `[B, M, N]` where `N = out_features`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let out = q4_matmul::<R, B>(x, &self.weights);
        match &self.bias {
            Some(bias) => out + bias.clone().unsqueeze::<3>(),
            None => out,
        }
    }
}

/// Fused Q/K/V projection: stores concatenated Q4 weights and splits output.
///
/// Instead of 3 separate Q4 matmul launches for wq, wk, wv, uses a single
/// concatenated weight matrix `[q_out + k_out + v_out, in_features]`.
/// Reduces kernel launches from 3 to 1 per layer.
pub struct Q4FusedQKV<R, B>
where
    R: CubeRuntime,
    B: Backend<FloatTensorPrimitive = CubeTensor<R>, Device = R::Device>,
{
    weights: Q4Tensor<R>,
    q_out: usize,
    k_out: usize,
    v_out: usize,
    _b: std::marker::PhantomData<B>,
}

/// Fused gate+up projection for SwiGLU: stores concatenated w1||w3 Q4 weights.
///
/// Reduces 2 Q4 matmul launches to 1 per FFN layer.
pub struct Q4FusedGateUp<R, B>
where
    R: CubeRuntime,
    B: Backend<FloatTensorPrimitive = CubeTensor<R>, Device = R::Device>,
{
    weights: Q4Tensor<R>,
    gate_out: usize,
    up_out: usize,
    _b: std::marker::PhantomData<B>,
}

impl<R, B> Q4FusedGateUp<R, B>
where
    R: CubeRuntime,
    B: Backend<FloatTensorPrimitive = CubeTensor<R>, Device = R::Device>,
{
    /// Create from a pre-built concatenated Q4 tensor.
    pub fn new(weights: Q4Tensor<R>, gate_out: usize, up_out: usize) -> Self {
        Self {
            weights,
            gate_out,
            up_out,
            _b: std::marker::PhantomData,
        }
    }

    /// Forward: single Q4 matmul → split into (gate, up).
    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let fused = q4_matmul::<R, B>(x, &self.weights);

        let gate = fused.clone().narrow(2, 0, self.gate_out);
        let up = fused.narrow(2, self.gate_out, self.up_out);

        (gate, up)
    }
}

impl<R, B> Q4FusedQKV<R, B>
where
    R: CubeRuntime,
    B: Backend<FloatTensorPrimitive = CubeTensor<R>, Device = R::Device>,
{
    /// Create from a pre-built concatenated Q4 tensor.
    pub fn new(weights: Q4Tensor<R>, q_out: usize, k_out: usize, v_out: usize) -> Self {
        Self {
            weights,
            q_out,
            k_out,
            v_out,
            _b: std::marker::PhantomData,
        }
    }

    /// Forward: single Q4 matmul → split into (q, k, v).
    ///
    /// `x` shape: `[B, M, K]`.
    /// Returns: `(q [B, M, q_out], k [B, M, k_out], v [B, M, v_out])`.
    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let fused = q4_matmul::<R, B>(x, &self.weights);

        let q = fused.clone().narrow(2, 0, self.q_out);
        let k = fused.clone().narrow(2, self.q_out, self.k_out);
        let v = fused.narrow(2, self.q_out + self.k_out, self.v_out);

        (q, k, v)
    }
}
