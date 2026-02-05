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

## Neighbor Graph Construction
- Builds a directed neighbor graph from positions and a radius.
- Supports uniform grid or KDTree backends and wrap-aware distances.
- Produces COO-style arrays for fast batched interaction kernels.

## Barnes-Hut Tree
- Approximates distant particle groups as a single mass.
- Good for reducing O(N^2) to roughly O(N log N) for N-body forces.
- Supports an optional numba backend for traversal.

## Two-level FMM (2D)
- Uses a uniform grid and far-field approximation by cell.
- Near-field interactions are computed directly for accuracy.

## Multi-level FMM (2D/3D)
- Builds a uniform quadtree/octree hierarchy.
- Computes monopole far-field contributions via interaction lists.
- Direct near-field interactions at the leaf level.

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

## Force Registry + Kernels
- Force registry manages multiple force families and per-species matrices.
- Packed arrays feed Numba kernels for batched force accumulation.
- Supports particle-life style forces plus Yukawa, Lennard-Jones, Morse, etc.

## Performance Notes
- All algorithms use contiguous float32 arrays.
- Heavy inner loops use vectorized numpy operations.
- For extreme scale, consider GPU or JIT backends.
