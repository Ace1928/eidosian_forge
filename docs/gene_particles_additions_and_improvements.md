# Gene Particles Additions, Enhancements, and TODOs

This document synthesizes improvements for Gene Particles based on:
- A deep review of PyParticles
- Performance profiling and data layout analysis
- External algorithm research (neighbor search, n-body, SPH, GPU pipelines)

## Learnings from PyParticles
1) **Numba kernels are a force multiplier**
   - PyParticles offloads core loops to Numba for massive speedups.
   - Gene Particles can migrate key CPU hot paths (neighbor evaluation, forces,
     reproduction, energy exchanges) into Numba kernels.

2) **Uniform grid + fixed neighborhood is fast and stable**
   - PyParticles uses a cell-linked list and local neighbor cell traversal.
   - Gene Particles can add a grid-based neighbor path for uniform density scenes
     as an alternative to KDTree.

3) **Layered force rules are expressive**
   - PyParticles stacks multiple force layers with independent radii and strengths.
   - Gene Particles could adopt multiple simultaneous force layers per type pair
     to build richer emergent dynamics without massive new code paths.

4) **GL instanced rendering is essential for scale**
   - PyParticles GL renderer decouples rendering from CPU and scales well.
   - Gene Particles can add a ModernGL renderer that accepts packed buffers for
     positions/energy and renders with shaders (glow, trails, depth cues).

5) **UI-first workflows accelerate iteration**
   - PyParticles UI enables live tuning (rules, species, physics).
   - Gene Particles should incorporate a live matrix editor for interaction rules,
     plus parameter sliders for reproduction, energy, and clustering.

## Immediate Additions (High Value)
1) **Hybrid neighbor path**
   - Add a uniform grid neighbor list for O(N) neighbor scans when interaction
     ranges are short relative to world size.
   - Provide automatic switching: KDTree for sparse long-range, grid for local.

2) **Numba acceleration**
   - Port the batched interaction evaluation (forces, predation, synergy) into a
     Numba kernel that consumes packed arrays and a neighbor list.

3) **GPU rendering pipeline**
   - Add `gene_particles/rendering/gl_renderer.py` using ModernGL.
   - Pack position/energy/type into a single VBO and render instanced quads.

4) **Rule stacking**
   - Allow multiple force layers per type pair (e.g., short-range repulsion +
     mid-range attraction + long-range gravity).

5) **Explicit 3D visualization**
   - Replace 2D projection hacks with a proper 3D camera model:
     - view/projection matrices
     - depth sorting or additive blending
     - optional orbit camera

## Medium-Term Enhancements
1) **Spatial hashing / Verlet lists**
   - Maintain neighbor lists with rebuild intervals for stable, dense regimes.
   - This can reduce per-step neighbor search cost substantially.

2) **Barnes–Hut / FMM for long-range**
   - For gravity-like long-range forces, add Barnes–Hut or FMM to reduce
     O(N^2) to O(N log N) or O(N).

3) **SPH / PBD module**
   - Add a Smoothed Particle Hydrodynamics or Position-Based Dynamics layer
     for fluid-like behaviors and cohesion stability.

4) **Energy field coupling**
   - A global energy field grid that diffuses over time, with particles sampling
     the field as an environmental driver.

5) **Genetic parameterization of force rules**
   - Allow genomes to encode interaction matrices and radii, not just behaviors.

## Data Structure Upgrades
- Introduce a global SoA buffer for all dynamic arrays:
  - pos, vel, energy, mass, age, type_id, state flags
- Allow per-type views into global buffers to avoid duplication.
- Support `float32` modes for performance-sensitive runs.

## Scalability & Parallelism
- Threaded job system for independent phases:
  - neighbor graph build
  - force evaluation
  - reproduction & death
- GPU compute kernels for physics (optional, via CuPy/numba.cuda).

## Aesthetic and Emergent Systems
- Trail rendering with decay buffers (2D/3D)
- Energy-driven bloom and color shifting
- Species-specific wave or oscillation mechanics
- Hierarchical attraction (predator/pack, prey/flocking, neutral drift)

## Testing & Coverage Plan (100%)
- Add unit tests for:
  - global neighbor graph correctness (pair-specific max_dist)
  - clustering results vs. reference implementation
  - energy conservation for predation/synergy
  - reproduction edge cases and mutation bounds
- Add integration tests:
  - deterministic small simulations with fixed seeds
- Add benchmarks with pinned configs to guard performance regressions

## Documentation Plan
- Expand `docs/gene_particles_reference.md` with:
  - data layout diagrams
  - force model equations
  - profiling results and optimization rationale
- Add a dedicated “Renderer Architecture” doc for GL path
- Add a “Genetic Systems” doc covering gene encoding and inheritance

## TODO Checklist
- [ ] Implement uniform grid neighbor path
- [ ] Add Numba kernels for interaction + clustering
- [ ] ModernGL renderer with instanced particles
- [ ] Rule stacking per type pair
- [ ] 3D camera and depth-correct rendering
- [ ] Barnes–Hut / FMM experimental branch
- [ ] SPH/PBD module
- [ ] Coverage + benchmark gating in CI
