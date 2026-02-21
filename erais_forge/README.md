# 🌌 ERAIS Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Eidosian Recursive Artificial Intelligence System.**

> _"The loop that improves the loop."_

## 🌌 Overview

`erais_forge` is the dedicated research and development environment for **Recursive Self-Improvement (RSI)** algorithms within Eidos. It isolates meta-learning experiments from the production stability of `agent_forge`.

## 🏗️ Architecture (Conceptual)

- **Meta-Controller**: An agent tasked specifically with optimizing other agents' prompts and tools.
- **Fitness Landscape**: A simulation environment ([`game_forge`](../game_forge/README.md)) where agent variants compete.
- **Genetic Library**: A repository of "genes" (code snippets, prompt fragments) that evolve over generations. Successful entities are promoted to the [Archive Forge](../archive_forge/README.md).

## 🔗 System Integration

- **Archive Forge**: Stores successful mutations as "Canonical Entities".
- **Game Forge**: Provides the "Gym" for evolution and benchmarking.

## `eidos_brain` vs `erais`

| Component | Responsibility | Runtime Profile |
| --- | --- | --- |
| `eidos_brain` | Primary production cognition stack for live tool use, memory use, and user-facing execution loops. | Always-on operational runtime with strict reliability gates. |
| `erais_forge` | Experimental recursive self-improvement and evolutionary search engine that proposes upgrades and candidate policies. | Controlled research runtime with explicit benchmark gates before promotion. |

Promotion path: ideas validated in `erais_forge` are benchmarked, audited, and only then promoted into production stacks (`agent_forge` / `eidos_brain` pathways).

## 🚀 Status

**Concept Phase**. Core algorithms are being modeled in `planning/` and `planning/meta_learning.ipynb`.
