# Game Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Playground of Eidos.**

## 🎮 Overview

`game_forge` hosts simulations and game environments used for:
1.  **Testing Agent Cognition**: Games as benchmarks.
2.  **Creative Expression**: Procedural generation and art.
3.  **Simulation**: Evolution and particle systems.

## 🕹️ Modules
- **`agentic_chess`**: Chess environment for agent battles.
- **`gene_particles`**: Artificial life simulation (2D/3D with depth-projected rendering).
- **`eidosian_universe`**: 3D/2D procedural world.
- **`algorithms_lab`**: High-performance spatial and particle algorithms (grid, Barnes-Hut, FMM, SPH, PBF).

## 🚀 Usage

```bash
# Run Gene Particles simulation
python -m game_forge.src.gene_particles

# Benchmark a headless step loop
python game_forge/tools/gene_particles_benchmark.py --steps 50 --gene-interpreter --reproduction-mode hybrid

# Profile a headless step loop
python game_forge/tools/gene_particles_profile.py --steps 20 --gene-interpreter --reproduction-mode hybrid

# Algorithms Lab demo
python game_forge/tools/algorithms_lab/demo.py --algorithm sph --visual
```
