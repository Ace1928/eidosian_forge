# Algorithms Lab

Algorithms Lab is a reusable collection of high-performance spatial and
particle simulation building blocks. The goal is to provide clean,
well-documented, and modular reference implementations that can be reused
by game_forge projects (including gene_particles) or copied into other
codebases with minimal dependency overhead.

## Highlights
- Uniform grid / spatial hash neighbor searches (2D/3D).
- Morton (Z-order) encoding for cache-friendly sorting.
- Spatial utilities (Morton order, SoA packing, adaptive cell sizing).
- Verlet neighbor lists (CSR format).
- Global neighbor graph construction utilities.
- Barnes-Hut tree for approximate N-body forces.
- Two-level FMM-style approximation (2D).
- Multi-level FMM-style solver (2D/3D).
- Force registry + batched multi-force kernels (particle-life style).
- SPH and PBF fluid solvers.
- XPBD (compliant) fluid solver.
- Optional numba and SciPy cKDTree acceleration.
- Optional GPU backends (OpenCL via PyOpenCL, CUDA/ROCm via CuPy).

## Installation
Algorithms Lab ships with the main game_forge package. The only required
runtime dependency is numpy. Visual demos optionally require pygame.
Optional accelerators:
- `numba` for JIT-accelerated neighbor enumeration.
- `scipy` for cKDTree-based neighbor search.

If installing via `pyproject.toml`, you can use:
```
pip install .[algorithms-lab]
```

## Quick Start

```python
import numpy as np
from algorithms_lab.core import Domain, WrapMode
from algorithms_lab.spatial_hash import UniformGrid

positions = np.random.rand(1024, 2).astype(np.float32)
domain = Domain(mins=np.array([0.0, 0.0]), maxs=np.array([1.0, 1.0]), wrap=WrapMode.WRAP)

grid = UniformGrid(domain, cell_size=0.05)
pairs_i, pairs_j = grid.neighbor_pairs(positions, radius=0.05)
print(pairs_i.shape, pairs_j.shape)
```

## CLI Demos

```bash
python game_forge/tools/algorithms_lab/demo.py --algorithm sph --visual
python game_forge/tools/algorithms_lab/demo.py --algorithm barnes-hut --visual
python game_forge/tools/algorithms_lab/benchmark.py --algorithms all
python game_forge/tools/algorithms_lab/demo.py --algorithm xpbd --neighbor-backend numba
python game_forge/tools/algorithms_lab/demo.py --algorithm fmm-ml --fmm-levels 4
python game_forge/tools/algorithms_lab/demo.py --algorithm barnes-hut --style modern --bh-backend numba
python game_forge/tools/algorithms_lab/demo.py --algorithm forces --visual
python game_forge/tools/algorithms_lab/profiler.py --algorithm barnes-hut
```

## Project Structure
- `game_forge/src/algorithms_lab/`: Core implementation modules.
- `game_forge/tools/algorithms_lab/`: Demos, benchmarks, and profilers.
- `game_forge/algorithms_lab/docs/`: Documentation and design notes.
- `game_forge/algorithms_lab/references/`: Sources and research summaries.

Additional docs:
- `game_forge/algorithms_lab/docs/PERFORMANCE.md`: Performance guidance and tuning.
- `game_forge/algorithms_lab/docs/GPU.md`: GPU backend usage.
- `game_forge/algorithms_lab/docs/FORCES.md`: Force registry and kernel usage.
