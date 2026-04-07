//! NVTX range annotations for CUDA profiling with Nsight Systems.
//!
//! All items are zero-cost no-ops when the `cuda` feature is disabled, so
//! annotations can be left in production code without overhead on non-CUDA builds.
//!
//! # Usage
//!
//! ```no_run
//! use voxtral_mini_realtime::nvtx_range;
//!
//! fn my_fn() {
//!     nvtx_range!("my_fn");          // range ends at end of block
//!     // ...
//!
//!     {
//!         nvtx_range!("inner_op");   // shorter nested range
//!         // ...
//!     }                              // "inner_op" pops here
//! }                                  // "my_fn" pops here
//! ```
//!
//! For dynamic labels:
//! ```no_run
//! # use voxtral_mini_realtime::nvtx_range;
//! # let m = 1usize; let n = 4096usize;
//! #[cfg(feature = "cuda")]
//! let _nvtx = voxtral_mini_realtime::nvtx::Guard::new(&format!("q4mm {m}x{n}"));
//! ```

/// RAII guard that pushes an NVTX range on creation and pops it on drop.
///
/// Prefer the [`nvtx_range!`] macro for static names. Use this type directly
/// for dynamically formatted labels.
#[cfg(feature = "cuda")]
pub struct Guard;

#[cfg(feature = "cuda")]
impl Guard {
    /// Push an NVTX range with the given name.
    #[inline]
    pub fn new(name: &str) -> Self {
        nvtx::range_push!("{}", name);
        Self
    }
}

#[cfg(feature = "cuda")]
impl Drop for Guard {
    #[inline]
    fn drop(&mut self) {
        nvtx::range_pop!();
    }
}

/// Push an NVTX range named `$name` for the rest of the enclosing scope.
///
/// The range is popped automatically when the binding `_nvtx` drops.
/// No-op (zero overhead) when the `cuda` feature is not enabled.
///
/// ```no_run
/// use voxtral_mini_realtime::nvtx_range;
/// nvtx_range!("my_operation");
/// ```
#[macro_export]
macro_rules! nvtx_range {
    ($name:expr) => {
        #[cfg(feature = "cuda")]
        let _nvtx = $crate::nvtx::Guard::new($name);
    };
}
