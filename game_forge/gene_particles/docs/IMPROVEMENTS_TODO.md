# Gene Particles: Additions, Enhancements, and TODOs

This document captures prioritized improvements to gene_particles based on
pyparticles analysis and algorithm_lab upgrades.

## A. High-Impact Performance Work
1. **Force Registry Integration**
   - Replace the single-rule interaction matrices with a multi-force registry.
   - Use packed arrays and a Numba kernel for batched force evaluation.
   - Keep a compatibility layer to interpret legacy rules as registry forces.

2. **Unified Global Neighbor Graph**
   - Build a wrap-aware neighbor graph once per frame.
   - Reuse graph across: interactions, predation, synergy, clustering.
   - Explore precomputed cell adjacency for incremental updates.

3. **Vectorized Energy Transfer**
   - Remove nested Python loops in `_apply_global_interactions` for predation/synergy.
   - Use pair-id grouping or sparse scatter-add to batch updates.

4. **Data Layout Upgrades**
   - Consolidate per-type arrays into SOA buffers with fixed capacity.
   - Keep per-type slices as lightweight views to reduce copying.

## B. Interaction and Behavior Enhancements
1. **Multi-Force Composition**
   - Allow layered forces (e.g., particle-life + Yukawa + Morse).
   - Enable per-force toggles in the UI.

2. **Advanced Flocking**
   - Implement Reynolds-style boids forces via the force registry
     or dedicated kernel, with configurable weights.

3. **Sexual Reproduction Pipeline**
   - Add crossover operators, mutation schedules, and trait heritability
     to the gene interpreter.

4. **Energy/Mass Transfer Modes**
   - Support configurable conservation laws and loss terms.
   - Add predation heuristics based on relative energy and proximity.

## C. Rendering and Visualization
1. **True 3D Rendering**
   - Move beyond 2D projection shading to actual 3D camera transforms.
   - Provide orbital controls and depth-based fog/occlusion.

2. **Visual Layering**
   - Add per-force visualization overlays (field lines, heatmaps).
   - Add neighbor graph debug rendering in 2D/3D.

## D. UX and Config
1. **Presets + Validation**
   - Add small/default/large presets like PyParticles.
   - Provide preflight validation and warnings on bad configs.

2. **Modular Config Packs**
   - Allow config JSON or TOML packs to define species, rules, and behaviors.
   - Enable hot-reload of configs with UI toggle.

## E. Testing and Benchmarking
1. **Force Kernel Validation**
   - Unit tests for each force family and registry packing.

2. **Integration Benchmarks**
   - Add a gene_particles benchmark suite with consistent seeds.

3. **Profiling Baselines**
   - Maintain baseline profiling outputs for regression detection.

## F. Documentation
- Add a dedicated `gene_particles/docs/` overview and architecture guide.
- Keep a running log of performance wins and profiling results.
