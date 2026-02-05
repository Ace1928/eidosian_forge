# Performance Priorities

Algorithms Lab is written to emphasize high throughput, predictable
memory usage, and scalability. This document captures performance
choices and guidance for extending the system.

## Design Principles
- **Contiguous float32 arrays**: All hot paths convert to float32 and
  contiguous storage to reduce cache misses.
- **CSR-style neighbor lists**: Neighbors are stored in dense integer
  arrays with offsets for fast sequential access.
- **Vectorized math**: Pairwise and per-particle operations are executed
  with numpy vectorization; Python loops only remain where unavoidable
  (cell pair iteration or tree traversal).
- **Cached domain sizes**: Domain sizes and their inverse are cached to
  avoid repeated allocation during minimal-image wrapping.
- **Auto backend selection**: Neighbor queries default to numba when
  available, falling back to numpy otherwise.

## Hot Path Notes
- `UniformGrid.neighbor_pairs` is cell-pair limited. Increasing `cell_size`
  reduces cell pairs but increases per-cell interactions; tuning matters.
- `NeighborList` rebuild thresholds are governed by `skin`. Larger skins
  reduce rebuilds but increase per-step neighbor processing.
- `BarnesHutTree` is dominated by tree traversal and minimal-image logic;
  for wrap-heavy simulations, consider reducing `theta` and enabling
  caching of subtrees in higher-level structures.
- `FMM2D` uses a two-level approximation; for large particle counts, a
  true multi-level FMM or GPU backend is recommended.

## Optional Accelerators
If you want further speedups, consider adding optional accelerators:
- **Numba**: JIT compile the loops in `UniformGrid.neighbor_pairs` and
  neighbor list builders. Enable by passing `backend=\"numba\"`.
- **SciPy cKDTree**: Use `KDTreeNeighborSearch` for fast radius queries
  in C, especially for irregular particle distributions.
- **CuPy/Numba CUDA**: Offload dense kernel operations for SPH/PBF.

These dependencies are intentionally optional to keep the core minimal.
