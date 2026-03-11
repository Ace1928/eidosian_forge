# 🎮 Computer Control Forge ⚡

```ascii
     ______ ____  __  _________  __ ______ __________     __________  ____  ____________
    / ____// __ \/  |/  / __ \ \/ // ____// ____/ __ \   / ____/ __ \/ __ \/ ____/ ____/
   / /    / / / / /|_/ / /_/ /\  // __/  / /   / / / /  / /_  / / / / /_/ / / __/ __/   
  / /___ / /_/ / /  / / ____/ / // /___ / /___/ /_/ /  / __/ / /_/ / _, _/ /_/ / /___   
  \____/ \____/_/  /_/_/     /_//_____/ \____/\____/  /_/    \____/_/ |_|\____/_____/   
                                                                                        
```

> _"Precise control, absolute safety, and multi-modal perception."_

## 🧠 Overview

`computer_control_forge` provides the "limbs" and "eyes" for Eidosian agents within the local execution environment. It enables keyboard/mouse control, Wayland/X11 integration, and rich screen perception capabilities. Designed with a **Safety-First** philosophy, it incorporates mandatory systemic kill switches and persistent telemetry pipelines for real-time sensorimotor modeling.

## ⚡ Current State & Metrics

- **Status**: 🟢 Elevated & Operational
- **Type**: Automation & Multimodal Perception
- **Test Coverage**: Safety and Kill-Switch logic verified (100%).
- **Inference Standard**: **Qwen 3.5 2B** (Used for OCR and visual reasoning).
- **Core Architecture**:
  - **`safety.py`**: Systemic fail-safes (file-based and signal-based kill switches) with Eidosian observability.
  - **`live_controller.py`**: Telemetry-emitting execution loop mapping actions to coordinates.
  - **`wayland_control.py`**: Specific adaptations for high-performance Wayland environments.
  - **`perception/`**: Multimodal analysis suite (OCR, screen-state diffing, and window management).

## 🛡️ Operational Safeguards

1. **Systemic Kill Switch**: Create `/tmp/eidosian_control_kill` to terminate all motor operations instantly.
2. **Deterministic Locking**: Operations are process-locked via `/tmp/eidosian_control.pid` to prevent concurrent actuator conflicts.
3. **Approval Gating**: High-risk actions (e.g., clicking external links) require explicit Agent Forge approval events.
4. **Rate Limiting**: Hardcoded delays ensure actions do not exceed human-operable velocities.

## 🚀 Usage & Workflows

### Systemic Safety Check

Always verify the integrity of the safety substrate before initializing actuators:
```python
from computer_control_forge.safety import is_kill_switch_active

if is_kill_switch_active():
    raise RuntimeError("Physical safety breach: Kill Switch is Engaged.")
```

### Sensorimotor Trace

Execute a controlled movement sequence with full telemetry streaming to the `event_bus`:
```python
from computer_control_forge.live_controller import LiveController, ControllerConfig

config = ControllerConfig(enable_agent_bus=True, enable_pipeline_trace=True)
with LiveController(config) as ctrl:
    ctrl.move_to(1920, 1080) # Move to corner
```

## 🔗 System Integration

- **Agent Forge**: Agents request "hands-on" missions via the workspace bus, which are then decomposed into motor steps.
- **Eidos MCP**: Exposed as the `computer_control` plugin, providing tools for screen capture and coordinate manipulation.
- **Glyph Forge**: Used to analyze captured screen buffers and transform them into semantic terminal maps.

## 🎯 Master Plan & Evolution

### Immediate Goals
- [x] Consolidate legacy safety documentation.
- [x] Integrate Eidosian observability decorators into the motor-control loop.
- [ ] Implement support for "Virtual Display" sandboxing to isolate agent clicks.

### Future Vector (Phase 3+)
- Build a "Visual Feedback Overlay" that renders the agent's intent (targets/paths) directly on screen.
- Transition to a "Predictive Perception" model that anticipates UI changes based on previous action results.
- Implement "Human-in-the-loop" visual approval requests for high-precision tasks.

---
*Generated and maintained by Eidos.*
