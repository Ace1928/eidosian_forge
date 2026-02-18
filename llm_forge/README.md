# 🤖 LLM Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Status: Evolving](https://img.shields.io/badge/Status-Evolving-orange.svg)](PLAN_EVOLUTION.md)

**The High-Performance Engine of Eidos.**

> _"Local intelligence, centralized fire."_

## 🤖 Overview

`llm_forge` is the centralized local inference and model management system for Eidos. It fully integrates and wraps **`llama.cpp`**, providing Pythonic interfaces for inference, benchmarking, quantization, and perplexity analysis.

It is designed to be the single source of local model truth for all other Forges.

## 🏗️ Architecture

- **Vendor Integration (`vendor/llama.cpp`)**: Contains the full source and locally optimized build of the `llama.cpp` engine.
- **Engine Layer (`src/llm_forge/engine/`)**:
    - `LocalCLIEngine`: Robust, single-turn inference via CLI.
    - `ServerEngine`: Persistent, high-concurrency API server.
    - `Quantizer`: Utilities for optimizing model weights (GGUF).
- **Benchmarking Suite (`src/llm_forge/benchmarking/`)**: 
    - Tools to measure throughput, latency, and model quality (perplexity).
- **Model Registry**: Centralized management of model paths and metadata.

## 🔗 System Integration

- **Documentation Forge**: Uses the `LocalCLIEngine` to power the Scribe.
- **Agent Forge**: Agents use the `ServerEngine` for low-latency tool calling.
- **ERAIS Forge**: Uses benchmarking data to evaluate model effectiveness during RSI.

## 🚀 Usage

### Inference (CLI Engine)

```python
from llm_forge.engine.local_cli import LocalCLIEngine, EngineConfig

config = EngineConfig(model_path="models/qwen-1.5b.gguf")
engine = LocalCLIEngine(config)

response = await engine.generate("Explain the Eidosian Framework.")
print(response)
```

### Benchmarking

```bash
# Run automated throughput sweep
python -m llm_forge.benchmarking.run_sweep --model models/qwen-1.5b.gguf
```

## 🛠️ Build Engine

To build or rebuild the underlying engine with local optimizations:

```bash
./llm_forge/scripts/build_engine.sh
```

---

**Version**: 0.2.0 (Evolutionary)
**Status**: Consolidating llama.cpp
**Maintainer**: EIDOS
