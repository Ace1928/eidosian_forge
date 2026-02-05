# Algorithms Lab

Algorithms Lab is a reusable collection of high-performance spatial and
particle simulation building blocks. The goal is to provide clean,
well-documented, and modular reference implementations that can be reused
by game_forge projects (including gene_particles) or copied into other
codebases with minimal dependency overhead.

## Highlights
- Uniform grid / spatial hash neighbor searches (2D/3D).
- Morton (Z-order) encoding for cache-friendly sorting.
- Verlet neighbor lists (CSR format).
- Barnes-Hut tree for approximate N-body forces.
- Two-level FMM-style approximation (2D).
- SPH and PBF fluid solvers.

## Installation
Algorithms Lab ships with the main game_forge package. The only required
runtime dependency is numpy. Visual demos optionally require pygame.

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
python game_forge/tools/algorithms_lab/profiler.py --algorithm barnes-hut
```

## Project Structure
- `game_forge/src/algorithms_lab/`: Core implementation modules.
- `game_forge/tools/algorithms_lab/`: Demos, benchmarks, and profilers.
- `game_forge/algorithms_lab/docs/`: Documentation and design notes.
- `game_forge/algorithms_lab/references/`: Sources and research summaries.

Additional docs:
- `game_forge/algorithms_lab/docs/PERFORMANCE.md`: Performance guidance and tuning.
