# ERAIS Forge Elevation Plan

**Objective**: Transform `erais_forge` from a concept into a functional Recursive Self-Improvement (RSI) engine.

## 🎯 Phase 1: The Genetic Substrate
- [x] ~~**Data Model (`src/erais_forge/models.py`)**: Define `Gene`, `Genome`, and `Generation` using Pydantic.~~
- [x] ~~**Genetic Library (`src/erais_forge/library.py`)**: Implement persistence layer.~~
- [x] ~~**Mutator (`src/erais_forge/mutator.py`)**: Logic to modify genes.~~
- [x] ~~**Controller (`src/erais_forge/controller.py`)**: The main loop (Select -> Mutate -> Evaluate -> Store).~~

## 🏋️ Phase 3: The Fitness Gym
- [ ] **Gym Adapter (`src/erais_forge/gym.py`)**: Interface to `game_forge`.
    - *Spec*: Run `agentic_chess` or `gene_particles` with a specific Genome.
    - *Metric*: Win rate or Survival time.

## 🔌 Phase 4: Integration & Observability
- [ ] **MCP Tools (`src/eidos_mcp/routers/erais.py`)**: `erais_evolve`, `erais_list_genes`.
- [ ] **CLI**: `erais evolve --generations 10`.
- [ ] **Tests**: >80% coverage.

## 🛡️ Constraints
- No uncontrolled code modification (User approval required for commit).
- Low resource footprint (runs in background).
