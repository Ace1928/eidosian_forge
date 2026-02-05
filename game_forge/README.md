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
- **`ECosmos`**: Evolving computational ecosystem simulation.
- **`gene_particles`**: Artificial life simulation (2D/3D with depth-projected rendering).
- **`eidosian_universe`**: 3D/2D procedural world.
- **`algorithms_lab`**: High-performance spatial and particle algorithms (grid, Barnes-Hut, FMM, SPH, PBF).
- **`Stratum`**: Layered physical simulation engine and scenario runner.
- **`falling_sand`**: Falling sand simulation and tooling.

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

## Launcher

```bash
python game_forge/tools/run.py --list
python game_forge/tools/run.py gene-particles
python game_forge/tools/run.py algorithms-lab-demo -- --algorithm sph --visual
python game_forge/tools/run.py eidosian-universe
python game_forge/tools/run.py stratum --list
```

```bash
PYTHONPATH=game_forge/src python -m algorithms_lab --demo -- --algorithm sph --visual
PYTHONPATH=game_forge/src python -m chess_game --list
```
