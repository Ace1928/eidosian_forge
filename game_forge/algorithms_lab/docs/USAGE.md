# Usage and Integration

Algorithms Lab is designed to be used directly from other game_forge
modules (including gene_particles) or copied into other repositories.

## Using as a Package

```python
import numpy as np
from algorithms_lab.core import Domain, WrapMode
from algorithms_lab.barnes_hut import BarnesHutTree
from algorithms_lab.kdtree import KDTreeNeighborSearch
from algorithms_lab.xpbd import XPBFSolver, XPBFState
from algorithms_lab.neighbors import NeighborSearch
from algorithms_lab.graph import build_neighbor_graph
from algorithms_lab.forces import ForceRegistry, accumulate_from_registry
from algorithms_lab.gpu import OpenCLNBody, CuPyNBody, HAS_PYOPENCL, HAS_CUPY
from algorithms_lab.fmm_multilevel import MultiLevelFMM

pos = np.random.rand(256, 2).astype(np.float32)
mass = np.ones(256, dtype=np.float32)

domain = Domain(mins=np.array([0.0, 0.0]), maxs=np.array([1.0, 1.0]), wrap=WrapMode.WRAP)

tree = BarnesHutTree(domain)
acc = tree.compute_acceleration(pos, mass)

# Optional: use cKDTree-backed neighbor queries
search = KDTreeNeighborSearch(domain)
pairs_i, pairs_j = search.neighbor_pairs(pos, radius=0.05)

# Unified neighbor search (auto-selects kdtree if available)
neighbors = NeighborSearch(domain, radius=0.05, method="auto", backend="numba")
pairs_i, pairs_j = neighbors.pairs(pos)

# Multi-level FMM acceleration
fmm = MultiLevelFMM(domain, levels=4)
acc = fmm.compute_acceleration(pos, mass)

# Force registry + neighbor graph (particle-life style)
registry = ForceRegistry(num_types=6)
graph = build_neighbor_graph(pos, radius=registry.get_max_radius(), domain=domain, method="grid", backend="numba")
type_ids = np.random.randint(0, 6, size=pos.shape[0], dtype=np.int32)
acc = accumulate_from_registry(pos, type_ids, graph.rows, graph.cols, registry, domain)

# GPU N-body acceleration (optional)
if HAS_PYOPENCL:
    acc = OpenCLNBody(domain).compute_acceleration(pos, mass)
if HAS_CUPY:
    acc = CuPyNBody(domain).compute_acceleration(pos, mass)

# Optional: XPBD fluid step
solver = XPBFSolver(domain, h=0.05, compliance=0.001)
state = XPBFState(positions=pos, velocities=np.zeros_like(pos), masses=mass)
state = solver.step(state)
```

## Using as Copy-Paste Code
- Each module is self-contained and depends only on numpy.
- Copy the file(s) you need plus `core.py`.

## Demos

```bash
# Visual demos
python game_forge/tools/algorithms_lab/demo.py --algorithm pbf --visual --style modern
python game_forge/tools/algorithms_lab/demo.py --algorithm forces --visual

# Headless benchmarks
python game_forge/tools/algorithms_lab/benchmark.py --algorithms all

# Profiling
python game_forge/tools/algorithms_lab/profiler.py --algorithm barnes-hut
```

## Notes
- Demos use pygame if installed; they fall back to headless mode.
- 3D simulations are supported; demos render the XY projection.
- For maximum speed, enable the numba backend for neighbor searches:
  `--neighbor-backend numba`.
- For Barnes-Hut, you can request the JIT backend:
  `--bh-backend numba`.

## Optional Dependencies
- `numba` enables JIT-accelerated neighbor enumeration.
- `scipy` enables cKDTree-based neighbor search.
