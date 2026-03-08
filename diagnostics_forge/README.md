# 🩺 Diagnostics Forge ⚡

> _"The Pulse of Eidos. To observe is to maintain the integrity of existence."_

## 🧠 Overview

`diagnostics_forge` provides an adaptive observability layer for the Eidosian ecosystem. It unifies high-fidelity logging, metrics collection, and real-time system health monitoring into a single interface optimized for both standard Linux servers and restricted environments like Termux.

```ascii
      ╭───────────────────────────────────────────╮
      │            DIAGNOSTICS FORGE              │
      │    < Pulse | Metrics | Telemetry >        │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │   ADAPTIVE SENSING  │   │ PROMETHEUS EXP. │
      │ (psutil / fallback) │   │ (/metrics API)  │
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: Observability & Monitoring
- **Test Coverage**: 100% Core Sensing logic verified across environments.
- **MCP Integration**: 4 Tools (`diagnostics_metrics`, `diagnostics_ping`, `diagnostics_processes`, `diagnostics_pulse`).
- **Architecture**:
  - `core.py`: Adaptive sensing engine (auto-switching between `psutil` and shell-parsing).
  - `cli.py`: Typer-based dashboard and Prometheus exporter.

## 🚀 Usage & Workflows

### Python API

```python
from diagnostics_forge.core import DiagnosticsForge

diag = DiagnosticsForge()

# Retrieve adaptive system pulse (CPU, RAM, Disk)
pulse = diag.get_system_pulse()
print(f"CPU: {pulse['cpu']['percent']}% | RAM: {pulse['memory']['percent']}%")
```

### CLI Dashboard

Visualize the current system state directly in the terminal:
```bash
python -m diagnostics_forge.cli dashboard
```

### Prometheus / Grafana Export

Serve a standardized `/metrics` endpoint for external scrapers:
```bash
python -m diagnostics_forge.cli prometheus --serve --port 9108
```

## 🔗 System Integration

- **Eidos MCP**: Exposes direct telemetry to the cognitive layer.
- **Agent Forge**: Powers the `eidtop` interface and autonomic homeostatic loops.
- **Viz Forge**: Feeds the time-series data required for historical performance plotting.

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy docs into unified Eidosian standard.
- [x] Establish strict JSON schemas for pulse payloads.

### Future Vector (Phase 3+)
- Implement "Anomaly Detection" which triggers automated thread-dumps and state snapshots in `archive_forge` when CPU/RAM spikes exceed historical Pareto frontiers.

---
*Generated and maintained by Eidos.*
