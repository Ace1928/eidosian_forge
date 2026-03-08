# 🎮 Game Forge ⚡

> _"The Playground of Eidos. Where simulations run, algorithms race, and agents learn."_

## 🧠 Overview

`game_forge` is the sandbox and simulation environment for Eidosian agents. It hosts a collection of isolated computational prototypes, artificial life simulations, and high-performance spatial algorithms. It serves as both a creative outlet and an empirical benchmark laboratory.

```ascii
      ╭───────────────────────────────────────────╮
      │               GAME FORGE                  │
      │    < Simulation | Benchmarks | Agents >   │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │ SPATIAL ALGORITHMS  │   │  A-LIFE/AGENTS  │
      │ (FMM, SPH, XPBD)    │   │ (GeneParticles) │
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Modules

- **Status**: 🟢 Elevated & Operational (Monorepo Collection)
- **Architecture**:
  - `agentic_chess`: Agent-versus-agent chess match runner (integrates with `agent_forge`).
  - `algorithms_lab`: High-performance spatial algorithms (Grid, Barnes-Hut, FMM, SPH, PBF).
  - `ECosmos`: Evolving computational ecosystem simulation.
  - `eidosian_universe`: 3D/2D procedural world.
  - `falling_sand`: Cellular automata falling sand simulation.
  - `gene_particles`: Artificial life simulation with depth-projected rendering.
  - `pyparticles`: Advanced particle physics engine.
  - `snake_ai_legacy`: Classic RL snake benchmark.
  - `Stratum`: Layered physical simulation engine.

*Note on Tests: Pytest must be run from within individual sub-project directories (e.g., `pyparticles`) due to isolated `PYTHONPATH` structures.*

## 🚀 Usage & Workflows

### Universal Launcher

```bash
# Ensure the forge virtualenv is active
source /home/lloyd/eidosian_forge/eidosian_venv/bin/activate

# List available runners
python game_forge/tools/run.py --list

# Run specific environments
python game_forge/tools/run.py gene-particles
python game_forge/tools/run.py algorithms-lab-demo -- --algorithm sph --visual
python game_forge/tools/run.py agentic-chess -- --white random --black agent-forge
```

### Benchmarks

`game_forge` includes extensive multi-environment benchmarking to track execution speeds across Termux limits:

```bash
# Quick regression benchmark
python game_forge/tools/run.py benchmark-suite -- --preset quick

# Headless specific simulation bench
python game_forge/tools/gene_particles_benchmark.py --steps 50 --gene-interpreter
```

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate fragmented documentation into a unified schema.
- [x] Stabilize benchmark suite collection.

### Future Vector (Phase 3+)
- Integrate `game_forge` directly as an Agent "Gym" via the MCP.
- Expand `agentic_chess` with local LLM evaluation loops to test cognitive persistence in agents.

---
*Generated and maintained by Eidos.*
