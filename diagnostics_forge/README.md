# Diagnostics Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**The Pulse of Eidos.**

## 🩺 Overview

`diagnostics_forge` handles the observability of the Eidosian System.
It provides:
- **Structured Logging**: JSON-enhanced logs for machine parsing.
- **Metrics**: Timer and counter aggregations.
- **Health Checks**: (Planned) System status verification.

## 🏗️ Architecture
- `diagnostics_core.py`: Main class `DiagnosticsForge` wrapping `logging` and internal metrics lists.

## 🚀 Usage

```python
from diagnostics_forge.diagnostics_core import DiagnosticsForge

diag = DiagnosticsForge(service_name="my_agent")
diag.log_event("INFO", "Agent started", agent_id="123")

timer = diag.start_timer("inference")
# ... do work ...
duration = diag.stop_timer(timer)
```