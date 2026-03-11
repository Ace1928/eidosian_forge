# 🩺 Diagnostics Forge ⚡

```ascii
     ____  _________   _______   ______  __________  _________  __________  ____  ____________
    / __ \/  _/   | | / / ___/ | / / __ \/ ___/_  __/  _/ ____/ / ____/ __ \/ __ \/ ____/ ____/
   / / / // // /| | |/ / (_ /  |/ / / / /\__ \ / /  / // /     / /_  / / / / /_/ / / __/ __/   
  / /_/ // // ___ | / / /__ / /|  / /_/ /___/ // /  / // /___  / __/ / /_/ / _, _/ /_/ / /___   
 /_____/___/_/  |_|/_/ \___/_/ |_/\____//____//_/  /___/\____/ /_/    \____/_/ |_|\____/_____/   
                                                                                                  
```

> _"The Pulse of Eidos. To observe is to maintain the integrity of existence."_

## 🧠 Overview

`diagnostics_forge` provides the adaptive observability layer for the Eidosian ecosystem. It unifies high-fidelity logging, metrics collection, and real-time system health monitoring into a single interface. Designed for seamless operation across standard Linux servers and restricted environments (Termux/Android), it enables agents to maintain interoceptive awareness of the physical and virtual substrates they inhabit.

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: Observability & Monitoring
- **Test Coverage**: 100% Core Sensing logic verified.
- **Inference Standard**: **Qwen 3.5 2B** (Used for automated log triage).
- **Core Architecture**:
  - **`core.py`**: Adaptive sensing engine utilizing `psutil` with a robust shell-parsing fallback for Android `/proc` restrictions with Eidosian observability.
  - **`telemetry/`**: Time-series collection and scalar aggregation for systemic "Vital Signs".
  - **`exporter/`**: Prometheus-compatible HTTP endpoint for external metrics scraping.
  - **`cli.py`**: Interactive dashboard and terminal-based performance monitoring.

## 🚀 Usage & Workflows

### Real-Time System Pulse

Retrieve a high-resolution snapshot of systemic load and saturation:
```python
from diagnostics_forge.core import DiagnosticsForge

diag = DiagnosticsForge()
pulse = diag.get_system_pulse()
print(f"Substrate Vitals: CPU {pulse['cpu']['percent']}% | RAM {pulse['memory']['percent']}%")
```

### Interactive Dashboard

Launch the Eidosian "Pulse" monitor directly in your terminal:
```bash
python -m diagnostics_forge.cli dashboard
```

### Prometheus Integration

Serve metrics for external visualization (e.g., Grafana):
```bash
python -m diagnostics_forge.cli prometheus --serve --port 9108
```

## 🔗 System Integration

- **Agent Forge**: Powers the **Homeostatic Autonomy Driver (HAD)** by supplying the scalar values for resource pressure drives.
- **Eidos MCP**: Exposes 4 specialized tools for remote system health auditing.
- **Viz Forge**: Feeds time-series data for historical performance plotting and Pareto frontier analysis.

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy metrics and documentation.
- [x] Implement adaptive sensing fallbacks for restricted Termux environments.
- [x] Integrate Eidosian observability decorators across all sensing loops.

### Future Vector (Phase 3+)
- Implement "Anomaly Detection" using local ML to trigger automated state snapshots when CPU/RAM spikes exceed historical norms.
- Add support for "Distributed Tracing" (OTLP) to track agent reasoning paths across multi-device networks.
- Build a "Self-Healing Actuator" that automatically clears caches or restarts services based on diagnostic thresholds.

---
*Generated and maintained by Eidos.*
