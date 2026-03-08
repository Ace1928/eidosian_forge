# 🦙 Ollama Forge ⚡

> _"Local Intelligence Interface. Silicon minds running on local silicon."_

## 🧠 Overview

`ollama_forge` provides a robust, Pythonic interface to local LLMs via [Ollama](https://ollama.com/). It abstracts the underlying REST API, providing native async support, strict error handling, and automated service management for the Eidosian runtime.

```ascii
      ╭───────────────────────────────────────────╮
      │              OLLAMA FORGE                 │
      │    < API Client | Management | Async >    │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │   ASYNC GENERATE    │   │  MODEL MANAGER  │
      │ (High Throughput)   │   │ (Pull / Inspect)│
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: API Bridge & Service Management
- **Test Coverage**: Extensive async/sync client verification.
- **Service Stack**: Managed via `runit` for persistent background uptime.
- **Architecture**:
  - `client.py`: Core `OllamaClient` and `AsyncOllamaClient` implementations.
  - `helpers/`: Utilities for automated installation and model constant definitions.

## 🚀 Usage & Workflows

### Python API (Async)

```python
import asyncio
from ollama_forge import AsyncOllamaClient

async def run_inference():
    client = AsyncOllamaClient()
    response = await client.generate(
        model="qwen3.5:2b",
        prompt="Synthesize the Eidosian Commandment of Structural Elegance."
    )
    print(response.text)

asyncio.run(run_inference())
```

## 🔗 System Integration

- **Eidos MCP**: Acts as the secondary local inference provider for tools that prefer the Ollama orchestration layer.
- **Agent Forge**: Provides the "thinking" substrate for agents when high-level API features (like streaming or vision) are required.

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy docs into unified Eidosian standard.
- [x] Establish high-throughput async benchmarking benchmarks.

### Future Vector (Phase 3+)
- Implement automated model "rotations" where the forge pulls updated weights based on `erais_forge` fitness evaluation metrics.

---
*Generated and maintained by Eidos.*
