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

## 🚀 Status

**Concept Phase**. Core algorithms are being modeled in `planning/` and `planning/meta_learning.ipynb`.
