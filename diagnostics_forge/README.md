# 🩺 Diagnostics Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Pulse of Eidos.**

## 🩺 Overview

`diagnostics_forge` provides the observability layer for the Eidosian ecosystem. It unifies logging, metrics collection, and health checks into a single interface.

## 🏗️ Architecture

- **Core (`core.py`)**: The `DiagnosticsForge` singleton.
- **Log Sink**: Structured logging (JSON/Console).
- **Metric Sink**: Aggregates timers, counters, and gauges.

## 🔗 System Integration

- **Agent Forge**: `eidosd` uses this for heartbeat metrics (`cpu`, `rss`, `load`).
- **Eidosian Core**: The `@eidosian` decorator pushes trace data here.

## 🚀 Usage

```python
from diagnostics_forge.core import DiagnosticsForge

diag = DiagnosticsForge(service_name="agent_core")

# Log event
diag.info("Agent started", version="1.0.0")

# Record metric
diag.gauge("memory_usage", 1024)

# Time block
with diag.timer("inference_latency"):
    run_inference()
```
