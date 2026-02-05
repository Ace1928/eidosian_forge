# Force Registry and Kernels

Algorithms Lab provides a reusable force registry and a batched, Numba-accelerated
kernel for evaluating multi-species interactions. The implementation mirrors the
"particle-life" style dynamics found in gene_particles and pyparticles, but
extends it with multiple force families and a packed data layout.

## Design Goals
- Allow multiple force families to be enabled simultaneously.
- Keep the force evaluation step vectorized and cache-friendly.
- Provide a compact pack format for reuse in other systems (GPU, JIT, etc.).

## Force Families
Supported force kernels include:
- Linear particle-life style attraction/repulsion.
- Inverse, inverse-square, and inverse-cube forces.
- Exponential, Gaussian, and Yukawa (screened) forces.
- Lennard-Jones and Morse bond-like potentials.

## Usage

```python
import numpy as np
from algorithms_lab.core import Domain, WrapMode
from algorithms_lab.graph import build_neighbor_graph
from algorithms_lab.forces import ForceRegistry, accumulate_from_registry

rng = np.random.default_rng(7)
positions = rng.random((512, 2), dtype=np.float32)
velocities = (rng.random((512, 2), dtype=np.float32) - 0.5) * 0.05

domain = Domain(
    mins=np.array([0.0, 0.0], dtype=np.float32),
    maxs=np.array([1.0, 1.0], dtype=np.float32),
    wrap=WrapMode.WRAP,
)

registry = ForceRegistry(num_types=6)
registry.randomize_all()

# Build a global neighbor graph and apply the packed forces.
graph = build_neighbor_graph(
    positions,
    radius=registry.get_max_radius(),
    domain=domain,
    method="grid",
    backend="numba",
)

type_ids = rng.integers(0, 6, size=positions.shape[0], dtype=np.int32)
acc = accumulate_from_registry(positions, type_ids, graph.rows, graph.cols, registry, domain)
velocities = velocities + acc * 0.01
positions = domain.wrap_positions(positions + velocities * 0.01)
```

## Notes
- For best performance, use the uniform-grid backend and enable numba.
- Force kernels are symmetric only if you include both edge directions.
- Use `ForceRegistry.pack()` if you want to reuse packed arrays across frames.
- Mass-weighted forces multiply the force magnitude by `mass_i * mass_j`
  (useful for gene_particles-style gravity).
