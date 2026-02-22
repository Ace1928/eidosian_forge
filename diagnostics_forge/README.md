# 🩺 Diagnostics Forge

[![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](../global_info.py)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Status: Operational](https://img.shields.io/badge/Status-Operational-green.svg)](README.md)

**The Pulse of Eidos.**

> _"To observe is to maintain the integrity of existence."_

## 🩺 Overview

`diagnostics_forge` provides an adaptive observability layer for the Eidosian ecosystem. It unifies logging, metrics collection, and system health monitoring into a single interface that remains functional across standard Linux servers and restricted environments like Termux.

## 🏗️ Architecture

- **Adaptive Core (`core.py`)**: Uses `psutil` for primary sensing and automatically falls back to shell-parsing (`top`, `uptime`) when system permissions are restricted.
- **Pulse Engine**: Aggregates CPU, RAM, Disk, and Load metrics.
- **Process Monitor**: Identifies and tracks Eidosian-related background tasks (LLMs, MCP servers, orchestrators).

## 🔗 System Integration

- **Eidos MCP**: Exposes `diagnostics_pulse` and `diagnostics_processes` tools.
- **Agent Forge**: Provides the telemetry data required for the `eidtop` TUI and daemon health checks.
- **Doc Forge**: Documented recursively to ensure observability schemas are always current.

## 🚀 Usage

### Python API

```python
from diagnostics_forge.core import DiagnosticsForge

diag = DiagnosticsForge()

# Get adaptive system pulse
pulse = diag.get_system_pulse()
print(f"CPU: {pulse['cpu']['percent']}% | RAM: {pulse['memory']['percent']}%")

# List Eidosian processes
for proc in diag.get_process_metrics():
    print(f"[{proc['pid']}] {proc['name']}: {proc['cpu']}% CPU")
```

### CLI Dashboard

```bash
# Render human-readable metrics dashboard
python -m diagnostics_forge.cli dashboard --metrics-file logs/core_metrics.json

# Emit raw JSON payload
python -m diagnostics_forge.cli --json dashboard --metrics-file logs/core_metrics.json
```

### Prometheus Export

```bash
# Print Prometheus exposition text once
python -m diagnostics_forge.cli prometheus

# Run /metrics endpoint
python -m diagnostics_forge.cli prometheus --serve --host 127.0.0.1 --port 9108
```

## 🛠️ Configuration

- **Environment Detection**: Automatically switches logic based on `PATH` inspection (Termux vs Linux).
- **Redirection Safety**: Logs are directed to `stderr` to maintain `stdout` protocol integrity for MCP communication.

## 🧪 Testing

```bash
# Run the test suite
pytest diagnostics_forge/tests/
```
