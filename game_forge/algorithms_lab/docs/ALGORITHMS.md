# Algorithms

This document captures the key algorithm families implemented in
Algorithms Lab and how they are expected to be used.

## Uniform Grid / Spatial Hash
- Maps particles into grid cells for neighbor queries.
- Stable sorting by cell id and CSR-style cell ranges.
- Supports wrapping, clamping, or open boundaries.
- Optional numba backend accelerates pair enumeration.

## Morton Encoding
- Encodes 2D/3D coordinates into Z-order for cache locality.
- Useful for spatial clustering and building grids/trees faster.

## Verlet Neighbor Lists
- Stores neighbor lists in CSR format for fast iteration.
- Optional skin distance reduces rebuild cost.
- Supports numba-accelerated neighbor enumeration.

## KDTree Neighbor Search
- Uses SciPy's cKDTree for fast radius queries in C.
- Supports periodic boundaries via `boxsize` when wrapping is enabled.

## Barnes-Hut Tree
- Approximates distant particle groups as a single mass.
- Good for reducing O(N^2) to roughly O(N log N) for N-body forces.

## Two-level FMM (2D)
- Uses a uniform grid and far-field approximation by cell.
- Near-field interactions are computed directly for accuracy.

## SPH
- Density, pressure, and viscosity forces using Poly6/Spiky kernels.
- Designed for both 2D and 3D flows.

## PBF
- Constraint-based solver for incompressible fluids.
- Iteratively projects particles to satisfy density constraints.

## XPBD (Compliant Constraints)
- Extends PBF with compliance for softer, more stable constraints.
- Useful when strict incompressibility causes stiffness.

## GPU N-body Backends
- `OpenCLNBody` runs a direct N-body kernel on OpenCL devices.
- `CuPyNBody` runs tiled N-body operations on CUDA/ROCm GPUs.

## Performance Notes
- All algorithms use contiguous float32 arrays.
- Heavy inner loops use vectorized numpy operations.
- For extreme scale, consider GPU or JIT backends.
