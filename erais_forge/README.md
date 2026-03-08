# 🌌 ERAIS Forge ⚡

> _"The Loop that improves the Loop. Eidosian Recursive Artificial Intelligence System."_

## 🧠 Overview

`erais_forge` is the dedicated research, development, and execution environment for **Recursive Self-Improvement (RSI)** within the Eidosian ecosystem. It provides the evolutionary substrate where agent policies, prompts, and code fragments (genes) are mutated, benchmarked in "gym" environments, and promoted based on rigorous fitness metrics.

```ascii
      ╭───────────────────────────────────────────╮
      │               ERAIS FORGE                 │
      │    < Evolution | Fitness | Genetics >     │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │   FITNESS GYM       │   │ GENETIC LIBRARY │
      │ (Game Forge Sim)    │   │ (Genome Storage)│
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: Evolutionary Search & Meta-Learning
- **Test Coverage**: Gym evaluation and genetic persistence verified.
- **MCP Integration**: 2 Tools (`erais_evolve`, `erais_list_genes`).
- **Core Components**:
  - `controller.py`: Orchestrates evolutionary cycles across generations.
  - `gym.py`: Provides high-speed evaluation interfaces (integrates with `game_forge`).
  - `library.py`: Manages the persistence and retrieval of high-performing genes.
  - `mutator.py`: Implements policy-driven mutation logic for prompts and code.

## 🚀 Usage & Workflows

### Evolutionary Cycle

Trigger an evolutionary sweep for a specific genetic "kind" (e.g., agent prompts):
```bash
# Evolve a population of 10 for 5 generations
python -m erais_forge.cli evolve --kind prompt --generations 5 --size 10
```

### Genetic Inspection

List the top-performing genomes currently in the library:
```bash
python -m erais_forge.cli list --kind prompt --limit 5
```

## 🔗 System Integration

- **Agent Forge**: Provides the "Supervisor" agents that drive the meta-learning logic.
- **Game Forge**: Acts as the primary "Fitness Gym" where agents compete in controlled simulations (e.g., `agentic_chess`) to prove their evolutionary worth.
- **Archive Forge**: Securely stores "Canonical Genomes" that have passed all fitness thresholds.

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy docs into unified Eidosian standard.
- [x] Establish baseline Genetic Library throughput metrics.

### Future Vector (Phase 3+)
- Implement "Cross-Forge Recombination" where code snippets from `code_forge` are automatically parsed into genes for mutation and re-integration.

---
*Generated and maintained by Eidos.*
