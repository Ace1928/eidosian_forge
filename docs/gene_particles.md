# Gene Particles

## Overview
Gene Particles is a cellular automata simulation built around genetic traits, emergent behaviors, and multi-type particle interactions. It combines vectorized physics, energy exchange, and gene-driven behavior to model evolution at scale in either 2D or 3D.

Core goals:
- Fast, vectorized updates for large particle counts
- Modular genetics and interpreter-driven behaviors
- Reproduction with inheritance, mutation, and speciation
- Extensible environment hooks and configuration

## Architecture
Modules live in `game_forge/src/gene_particles`.

Key modules:
- `gp_main.py`: CLI entrypoint and simulation bootstrap
- `gp_config.py`: Simulation and genetics configuration
- `gp_automata.py`: Main loop, physics integration, and orchestrator
- `gp_interpreter.py`: Genetic interpreter and gene routing
- `gp_genes.py`: Gene application and reproduction mechanics
- `gp_manager.py`: Manager-based reproduction and population control
- `gp_types.py`: Data structures and type definitions
- `gp_rules.py`: Interaction rule generation and evolution
- `gp_renderer.py`: Visualization and statistics overlay
- `gp_utility.py`: Shared physics and mutation helpers

## Spatial Model (2D/3D)
The simulation runs in 2D or 3D with a unified data layout. In 3D, positions and velocities include `z`/`vz`, and the renderer projects the volume onto the 2D screen.

Key configuration fields:
- `spatial_dimensions`: `2` or `3`
- `world_depth`: depth of the simulation volume (3D only)
- `projection_mode`: `orthographic` or `perspective`
- `projection_distance`: camera distance for perspective
- `depth_fade_strength`: depth-based brightness attenuation
- `depth_min_scale`, `depth_max_scale`: projection scale clamps

## Environment Hooks
`SimulationConfig` exposes environment parameters used in gene expression:
- `day_night_cycle`, `day_length`, `time`, `time_step`
- `temperature`, `temperature_drift`, `temperature_noise`, `temperature_bounds`
- `environment_hooks`: callables invoked each frame

The main loop calls `config.advance_environment(frame)` once per frame to advance time, update temperature, and run hooks.

## Genetic Interpreter
The interpreter decodes a gene sequence into behavior each tick:
- Movement: velocity updates and energy cost
- Interaction: local attraction/repulsion
- Energy: passive gain and metabolic decay
- Growth: size and maturity
- Reproduction: sexual or asexual pipeline
- Predation: target selection and energy transfer

Interpreter controls:
- `use_gene_interpreter`: enable interpreter updates
- `gene_interpreter_interval`: apply every N frames
- `gene_sequence`: optional explicit gene sequence

## Reproduction Pipeline
There are three reproduction modes:
- `manager`: manager-driven, asexual, vectorized reproduction
- `genes`: reproduction via interpreter gene expression
- `hybrid`: both manager and gene-driven reproduction

### Sexual reproduction
Sexual reproduction includes:
- Mate selection within radius using a selectable strategy
- Genetic compatibility gating and speciation logic
- Multiple crossover modes: uniform, arithmetic, blend, segment
- Linkage groups to preserve trait coupling
- Post-crossover jitter and mutation

Key configuration fields:
- `sexual_reproduction_enabled`, `sexual_reproduction_probability`
- `mate_selection_radius`, `mate_selection_max_neighbors`
- `mate_selection_strategy`: random, energy, compatibility, hybrid
- `compatibility_threshold`, `compatibility_weight`, `energy_weight`
- `crossover_mode_weights`, `crossover_blend_alpha`, `crossover_jitter`
- `recombination_rate`, `linkage_groups`
- `allow_cross_type_mating`: allow cross-species pairing within a type
- `allow_hybridization`, `hybridization_cost`

### Asexual reproduction
Asexual reproduction uses trait mutation with a vectorized path:
- Parent energy is reduced by reproduction cost and offspring allocation
- Offspring traits are mutated and clamped to valid ranges
- Speciation is triggered by genetic distance

## Performance Notes
- Interaction physics and reproduction use vectorized NumPy operations
- `KDTree` accelerates neighbor queries for clustering and mating
- Bulk append routines (`add_components_bulk`) reduce O(n^2) growth

## Testing
Gene Particles has dedicated unit tests under `game_forge/tests`:
- `test_gene_particles_*` files cover interpreter, genes, config, automata
- Coverage target: 100% for `game_forge/src/gene_particles`

Run tests:
```
pytest -q game_forge/tests/test_gene_particles_*.py \
  --cov=game_forge/src/gene_particles --cov-report=term-missing
```

## Benchmarking and Profiling
Headless tools:
- `game_forge/tools/gene_particles_benchmark.py`
- `game_forge/tools/gene_particles_profile.py`

Examples:
```
python game_forge/tools/gene_particles_benchmark.py --steps 50 --gene-interpreter --reproduction-mode hybrid
python game_forge/tools/gene_particles_benchmark.py --steps 50 --dimensions 3 --world-depth 800
python game_forge/tools/gene_particles_profile.py --steps 20 --gene-interpreter --reproduction-mode hybrid
python game_forge/tools/gene_particles_profile.py --steps 20 --dimensions 2
```

## Known Constraints
- Rendering is per-particle; large populations are CPU-bound
- KDTree requires SciPy for best performance (fallbacks are no-op)
- Energy efficiency can be negative; adjust ranges if needed
