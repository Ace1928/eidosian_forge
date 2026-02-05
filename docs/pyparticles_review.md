# PyParticles Review (Comprehensive)

## Summary
PyParticles is a fast, interactive particle simulation with a strong bias toward
GPU-friendly rendering and a highly optimized CPU physics core. The system
combines Numba-accelerated kernels, a spatial grid for neighborhood queries, and
an extensible UI for live parameter editing. The newer OpenGL renderer uses
instanced rendering and shader-based particles for large counts. The codebase is
well-structured for iteration speed and visual experimentation.

Key themes:
- CPU physics built on Numba kernels with a uniform spatial grid.
- Global species configuration (per-type parameters) rather than per-particle genomes.
- Biological layer: energy, metabolism, death, and mitosis.
- OpenGL path that decouples rendering from the CPU in a scalable way.
- UI-first workflow for rapid tuning of rules and species parameters.

## Module Map
Location: `game_forge/pyparticles/src/pyparticles`

- `app.py`: main loop, GL vs SDL renderer selection, UI bootstrapping.
- `core/types.py`: dataclasses for config, rules, species parameters, and particle state.
- `physics/engine.py`: simulation orchestration, biology, grid build, force computation.
- `physics/kernels.py`: Numba kernels for grid fill, forces, biology, and integration.
- `rendering/canvas.py`: SDL-based renderer (sprites/pixels/waves).
- `rendering/gl_renderer.py`: ModernGL instanced renderer + UI overlay.
- `ui/gui.py`: pygame_gui-based UI with sliders, matrix editor, and persistence.

## Core Data Model
`ParticleState` is a contiguous SoA layout:
- `pos: (N,2) float32`
- `vel: (N,2) float32`
- `colors: (N,) int32` (species index)
- `angle: (N,) float32`
- `ang_vel: (N,) float32`
- `energy: (N,) float32`
- `active: int` (current active count)

`SpeciesConfig` is a per-type parameter store (radius, wave params, metabolism,
mitosis threshold, start energy). There is no per-particle genome; all biology
is species-scoped.

`InteractionRule` defines force models with:
- `force_type` (linear, inverse-square, inverse-cube, repel-only)
- `matrix` (T x T interaction coefficients)
- `min_radius`, `max_radius`, `strength`, `softening`, `enabled`

## Physics Pipeline
Primary frame step in `PhysicsEngine.update`:
1) **Biology update** (metabolism, death, split):
   - `update_biology` kernel produces flags (alive/dead/split).
   - Engine compacts survivors and spawns new particles for splits.
2) **Spatial grid**:
   - `fill_grid` builds a uniform grid (cell list) based on `cell_size`.
3) **Force computation**:
   - `compute_forces_multi` runs over particles in parallel and visits neighbor
     cells (3x3) to compute multi-layer force rules plus wave mechanics.
4) **Integration**:
   - `integrate` updates velocities/positions, applies friction, and clamps
     to bounds with damping.

### Biology Layer
- Energy changes: `ambient_energy_rate - metabolism` per step.
- Death when energy <= 0; split when energy >= threshold.
- Mitosis uses copy-and-offset with energy split; no mutation yet.

### Spatial Hashing / Grid
- Grid size based on maximum interaction radius.
- `grid_counts` stores per-cell counts; `grid_cells` stores indices.
- Neighborhood interaction uses 3x3 local cell blocks.

## Force Model Details
`compute_forces_multi` supports multiple stacked rule layers:
- Linear “particle-life” force
- Inverse-square gravity-like force
- Inverse-cube strong force
- Repel-only force mode

Each rule has its own max/min radius, strength, and softening. Force direction
is based on normalized displacement. Forces are accumulated per particle.

### Wave Mechanics
Wave interaction is a local radial boundary model:
- Each particle has a base radius + angular wave perturbation.
- When waves overlap (gap < 0), a repulsive force and torque are applied.
- This adds rotational dynamics (angle + angular velocity) to the system.

## Rendering
### SDL Canvas (`rendering/canvas.py`)
- Sprite mode for moderate counts; PixelArray for large counts.
- Wave mode draws orientation lines for low particle counts (debug view).
- Uses HSV palette with per-type colors.

### OpenGL Renderer (`rendering/gl_renderer.py`)
- ModernGL context with instanced point rendering and geometry shader.
- VBO packed with per-particle data:
  - position, color, radius/freq/amp, angle, energy
- Shader-driven particle size and glow effects.
- UI is composited via a texture pipeline, enabling overlays.

## UI
`ui/gui.py` offers a comprehensive live-tuning interface:
- Particle count, species count controls
- Rule toggles and rule matrix editor
- Sliders for friction, dt, radius, strength
- Save/load matrix presets

This is a major usability strength: immediate feedback and rapid exploration.

## Performance Strengths
- Numba kernels for tight loops and neighbor evaluation.
- Uniform grid limits neighbor lookups to local cells.
- Instanced OpenGL rendering for large particle counts.

## Weak Spots / Risks
- Biology split logic is not fully clean (commented uncertainty around compaction
  ordering and split index remapping).
- Split/mutation logic is species-level only; no per-particle genetics.
- Wave rendering in SDL path is intentionally slow; debug-only.
- Grid is fixed-size with max_per_cell; overflow is silent.

## Test Coverage
Current tests cover core, physics, rendering, and UI modules. There is room to
add tests for:
- Mitosis compaction invariants
- Grid overflow behavior
- Rule stacking correctness
- GL renderer data packing integrity

## Opportunities / Lessons
1) **Spatial grid with bounded neighbor queries** is highly effective for local
   interaction models.
2) **Numba kernels** provide near-C speed for pure Python simulations.
3) **GPU render pipelines** allow scaling visuals independently of CPU physics.
4) **Rule stacking** is a powerful way to combine distinct force regimes.
5) **Rich UI tooling** accelerates iteration and parameter discovery.
