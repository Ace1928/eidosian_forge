# 🤖 LLM Forge ⚡

> _"The Engine of Eidos. Local intelligence, centralized fire."_

## 🧠 Overview

`llm_forge` is the central local inference and model management infrastructure for the Eidosian ecosystem. It fully integrates and optimizes **`llama.cpp`**, providing high-performance Pythonic interfaces for local generation, structural benchmarking, GGUF quantization, and perplexity analysis.

```ascii
      ╭───────────────────────────────────────────╮
      │               LLM FORGE                   │
      │    < Inference | Benchmarking | Quant >   │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │   LLAMA.CPP CORE    │   │  CANONICAL MODEL│
      │ (Local C++ Engine)  │   │ (Qwen 3.5 2B) │
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Evolving
- **Type**: Local Inference Engine
- **Canonical Model**: `Qwen3.5-2B-Instruct-Q8_0.gguf`
- **Test Coverage**: Core engine wrappers and provider logic verified.
- **MCP Integration**: 2 Tools (`llm_local_generate`, `llm_run_benchmark`).
- **Architecture**:
  - `engine/local_cli.py`: Robust, single-turn inference via optimized CLI calls.
  - `benchmarking/`: Automated throughput sweeps and latency tracking.
  - `providers/`: Support for `ollama` and `openai` (local-proxy) compatibility.

## 🚀 Usage & Workflows

### Python API (CLI Engine)

```python
from llm_forge.engine.local_cli import LocalCLIEngine, EngineConfig

# Point to the canonical Eidosian model
config = EngineConfig(model_path="/home/lloyd/eidosian_forge/models/Qwen3.5-2B-Instruct-Q8_0.gguf")
engine = LocalCLIEngine(config)

response = await engine.generate("Clarify the Eidosian Operational Principles.")
print(response)
```

### Benchmarking Suite

Execute a comprehensive performance sweep to establish baseline TPS (Tokens Per Second) for the current environment:
```bash
python -m llm_forge.benchmarking.run_sweep --model /home/lloyd/eidosian_forge/models/Qwen3.5-2B-Instruct-Q8_0.gguf
```

## 🛠️ Build & Optimization

To rebuild the underlying C++ engine with local Termux optimizations:
```bash
./llm_forge/scripts/build_engine.sh
```

---
*Generated and maintained by Eidos.*
