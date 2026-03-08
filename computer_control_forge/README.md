# 🎮 Computer Control Forge ⚡

> _"Precise control, absolute safety, multimodal perception."_

## 🧠 Overview

`computer_control_forge` provides local keyboard/mouse control, Wayland integration, and rich screen perception capabilities for Eidosian agents. It is designed with mandatory safety mechanisms (kill switches) and telemetry pipelines for real-time sensorimotor modeling.

```ascii
      ╭───────────────────────────────────────────╮
      │         COMPUTER CONTROL FORGE            │
      │    < Perception | Execution | Safety >    │
      ╰──────────┬─────────────────────┬──────────╯
                 │                     │
      ╭──────────┴──────────╮   ╭──────┴──────────╮
      │ SENSORIMOTOR TRACE  │   │  KILL SWITCH    │
      │ (Agent Workspace)   │   │ (/tmp/pid)      │
      ╰─────────────────────╯   ╰─────────────────╯
```

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: Automation & Perception
- **Test Coverage**: Safety logic verified (100% pass on kill-switches).
- **Core Modules**:
  - `safety.py`: Mandatory file/keyboard kill switches.
  - `live_controller.py`: Telemetry-emitting execution loop.
  - `wayland_control.py`: Specific adaptations for Wayland.
  - `multimodal_perception.py`: OCR and screen state analysis.

## 🛡️ Safety First Design

1. **Kill Switch**: Multiple termination mechanisms:
   - File-based: Create `/tmp/eidosian_control_kill` to stop immediately.
   - Process: `kill -9 $(cat /tmp/eidosian_control.pid)` to force stop.
2. **Explicit Activation**: Control never auto-starts.
3. **Local Only**: No network control capabilities by design.
4. **Rate Limiting**: Built-in delays prevent runaway actions.

## 🚀 Usage

### Safe Execution

```python
from computer_control_forge.safety import KillSwitch

# File-based safety check
if KillSwitch.is_active():
    print("Operations blocked by Kill Switch.")
```

### Live Telemetry (Sensorimotor Pipeline)

The live controller emits a structured sensorimotor trace and forwards it to the Agent Forge workspace bus:

```python
from computer_control_forge.live_controller import LiveController, ControllerConfig

config = ControllerConfig(enable_agent_bus=True, enable_pipeline_trace=True)
with LiveController(config) as ctrl:
    ctrl.move_to(960, 540)
```

CLI Trace Example:

```bash
python -m computer_control_forge.cli.main state \
  --agent-bus \
  --agent-bus-dir /home/lloyd/eidosian_forge/agent_forge/state \
  --pipeline-trace
```

## 🔗 System Integration

- **Eidos MCP Plugin**: Exposed as the `computer_control` plugin.
- **Agent Forge**: Agents can request "hands-on" tasks via the workspace bus.

---
*Generated and maintained by Eidos.*
