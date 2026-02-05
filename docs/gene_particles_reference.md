# Gene Particles Reference

This document provides a full, granular reference for Gene Particles. It is
intended to be an exhaustive companion to the codebase, covering the data
structures, algorithms, and interactions that make up the simulation.

Contents:
- Architecture and module map
- Frame lifecycle and data flow
- Global particle view + neighbor graph
- Interaction physics and energy transfers
- Clustering (boids) system
- Reproduction and genetics
- Environment hooks and config parameters
- Rendering and UI
- Performance model and profiling guidance

## Architecture Overview
Gene Particles lives in `game_forge/src/gene_particles` and is composed of:
- `gp_main.py`: CLI entrypoint and simulation bootstrap
- `gp_config.py`: simulation parameters and validation
- `gp_automata.py`: main loop, interactions, integration, and orchestration
- `gp_rules.py`: interaction rule generation and evolution
- `gp_interpreter.py`: gene decoding and behavior application
- `gp_genes.py`: gene application functions and reproduction logic
- `gp_manager.py`: manager-mode reproduction and population control
- `gp_types.py`: per-type data containers and helper utilities
- `gp_renderer.py`: particle rendering and stats overlay
- `gp_ui.py`: GUI overlay and input handling
- `gp_utility.py`: shared physics, mutation, and math helpers

Each module is designed to be independently testable, with data movement between
modules minimized and structured via typed NumPy arrays.

## Frame Lifecycle (High-Level)
Per simulation frame, the core loop performs:
1) Environment advance: `SimulationConfig.advance_environment(frame)`
2) Interaction evolution: `InteractionRules.evolve_parameters(frame)`
3) Batched inter-type interaction computation (global neighbor graph)
4) Gene interpreter update (optional, at interval)
5) Clustering (boids) update using global neighbor graph
6) Reproduction: manager- or gene-based, depending on mode
7) Death handling and population cleanup
8) Rendering + UI

The benchmark/profile scripts reproduce this path without rendering.

## Global Particle View
`gp_automata.CellularAutomata` builds a packed, contiguous view of all particles
in `_build_global_view` to reduce per-type loops and allow batched operations.

Global arrays (all length `total_particles`):
- `positions`: NxD (D=2 or 3)
- `velocities`: NxD
- `energy`: N
- `mass`: N (0.0 for non-mass types)
- `alive`: N (bool)
- `type_ids`: N (int)
- `offsets`: (n_types+1), prefix sums of per-type counts
- `counts`: per-type counts

The global view is used for the batched interaction and clustering paths. After
interactions, energy/mass and velocities are scattered back into per-type arrays
using the stored offsets.

## Neighbor Graph Pipeline
There are two construction paths:

1) **Per-pair graph (preferred)**
   - Uses per-type `KDTree`s built in `_build_interaction_cache`.
   - For each type pair (i, j), a radius query uses the specific `max_dist[i, j]`.
   - Resulting per-pair edges are concatenated into global arrays.
   - Graph is tagged `filtered_by_max_dist=True` to avoid redundant filtering.

2) **Global graph (fallback)**
   - Builds a single KDTree over all particles with `max_dist_global`.
   - Produces a global radius graph, then filters per-edge with the pairwise
     `max_dist` matrix.

Graph outputs:
- `rows`, `cols`: edge endpoints in global index space
- `dist`: Euclidean distances for each edge
- wrap info (`wrap_mode`, `world_size`, `inv_world_size`) for minimal-image deltas

### Wrap Mode
When `boundary_mode == "wrap"`, KDTree queries use periodic distances if
supported (`boxsize=world_size`) and delta computations use minimal image
wrapping via `wrap_deltas`.

## Interaction Physics
Interactions are computed in `_apply_global_interactions` using the global graph.

For each edge `(i, j)`:
- Relative displacement `dx, dy, dz` computed from positions (with optional wrap)
- `dist`, `inv_dist`, `inv_dist_sq` derived from edge distance

### Potential Force
For pair types (A, B):
```
F_pot = potential_strength[A,B] * (1 / r^2)
fx += F_pot * dx
fy += F_pot * dy
fz += F_pot * dz
```
Potential can be attractive or repulsive depending on sign.

### Gravity Force
If both types are mass-based and `use_gravity[A,B]`:
```
F_grav = gravity_factor[A,B] * (m_a * m_b) / r^3
fx -= F_grav * dx
fy -= F_grav * dy
fz -= F_grav * dz
```

### Predation (Give/Take)
If `give_take_matrix[A,B]` and within `predation_range`:
- Receiver (B) loses energy; giver (A) gains energy
- Optional mass transfer mirrors energy transfer when enabled

### Synergy
If `synergy_matrix[A,B] > 0` and within `synergy_range`:
- Energy of the two particles moves toward their average, weighted by synergy

All force contributions are accumulated using `np.bincount` for efficient
per-particle aggregation.

## Clustering (Boids)
Clustering uses a global neighbor graph limited to `cluster_radius`:
1) Keep only edges where both endpoints share the same type and neighbor is alive
2) Compute per-particle neighbor mean velocity and center
3) Alignment, cohesion, separation are computed and summed:
```
alignment = (avg_vel - vel) * alignment_strength
cohesion  = (center - pos) * cohesion_strength
separation = (pos - center) * separation_strength
```
4) Add total force to velocity

## Reproduction
Two modes:
- **Manager mode** (`ReproductionMode.MANAGER`): bulk reproduction in `gp_manager`
- **Gene mode** (`ReproductionMode.GENES`): gene-driven reproduction in interpreter
- **Hybrid**: both paths are active

### Asexual
- Parent energy is reduced by reproduction cost
- Offspring traits are mutated and clamped
- Speciation is triggered by genetic distance

### Sexual
- Mate selection within a radius
- Compatibility gating
- Multiple crossover modes (uniform, arithmetic, blend, segment)
- Mutation applied post-crossover
- Optional hybridization and cross-type mating (config-driven)

## Genetic Interpreter
`gp_interpreter` decodes gene sequences into concrete behaviors:
- movement and acceleration
- energy usage and gain
- interaction affinity modulation
- growth and maturity
- reproduction triggers

Interpreter cadence is controlled by `gene_interpreter_interval`.

## Environment Hooks
`SimulationConfig` includes environment parameters used by genes:
- time / day-night cycle
- temperature and drift
- `environment_hooks`: callables executed each frame

## Rendering and UI
Rendering happens in `gp_renderer`:
- Particles rendered as circles with energy-based brightness
- 3D positions projected to 2D via orthographic or perspective projection
- Depth-based fade and scale controls

UI overlay in `gp_ui` provides:
- status panel, config panel, and controls
- key bindings for common toggles (pause, stats, projection, size, etc.)

## Performance Model
Key performance features:
- Packed global view for contiguous data access
- Batched interaction evaluation using `np.bincount`
- Pairwise neighbor graphs with KDTree acceleration
- Wrap handling optimized with precomputed inverse world sizes

Primary hot spots (as measured in profiling):
- `_apply_global_interactions`
- `_build_global_neighbor_graph_pairwise`
- `wrap_deltas` for wrap-heavy workloads

## Testing and Profiling
Tests:
```
pytest -q game_forge/tests/test_gene_particles_*.py \
  --cov=game_forge/src/gene_particles --cov-report=term-missing
```

Benchmark:
```
python game_forge/tools/gene_particles_benchmark.py --steps 1000 --cell-types 8 --particles 500
```

Profile:
```
python game_forge/tools/gene_particles_profile.py --steps 1000 --top 50
```

## Notes and Assumptions
- When `SDL_VIDEODRIVER=dummy`, rendering is disabled but simulation still runs.
- The simulation is CPU-bound for very large particle counts.
- SciPy is recommended for KDTree acceleration.
