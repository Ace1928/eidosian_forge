# 🎮 Computer Control Forge

> _"Precise control, absolute safety, full transparency."_

## Purpose

Provide local keyboard/mouse control and screen capture capabilities for Eidosian agents with mandatory safety mechanisms.

## Safety First Design

This module is built with safety as the primary concern:

1. **Kill Switch**: Multiple kill mechanisms:
   - File-based: Create `/tmp/eidosian_control_kill` to stop immediately
   - Keyboard: Press `Ctrl+Alt+Shift+K` to halt all control
   - Process: `kill -9 $(cat /tmp/eidosian_control.pid)` to force stop
   
2. **Explicit Activation**: Control never auto-starts; requires explicit `start()` call

3. **Full Audit Trail**: Every action logged with timestamp and provenance

4. **Local Only**: No network control capabilities by design

5. **Rate Limiting**: Built-in delays prevent runaway actions

## Installation

```bash
# Install system dependencies (one-time)
sudo apt install xdotool scrot

# Install Python dependencies
cd /home/lloyd/eidosian_forge
source eidosian_venv/bin/activate
pip install pynput pillow mss

# Test installation
python -m computer_control_forge.test_safety
```

## Usage

```python
from computer_control_forge import ControlService, KillSwitch

# Initialize with safety checks
control = ControlService(
    require_kill_switch=True,
    rate_limit_ms=100,  # Min 100ms between actions
    log_dir="./control_logs"
)

# Activate (will fail if kill file exists)
control.start()

# Perform actions
control.type_text("Hello, World!")
control.click(100, 200)
screenshot = control.capture_screen()

# Always stop when done
control.stop()
```

### Live telemetry (sensorimotor pipeline)

The live controller can emit a structured sensorimotor trace and forward it to the
Agent Forge workspace bus:

```python
from computer_control_forge.live_controller import LiveController, ControllerConfig

config = ControllerConfig(enable_agent_bus=True, enable_pipeline_trace=True)
with LiveController(config) as ctrl:
    ctrl.move_to(960, 540)
```

CLI trace example:

```bash
python -m computer_control_forge.live_controller state \
  --agent-bus \
  --agent-bus-dir /home/lloyd/eidosian_forge/agent_forge/state \
  --pipeline-trace \
  --no-file
```

## Kill Switch Commands

```bash
# Create kill file (stops all control immediately)
touch /tmp/eidosian_control_kill

# Remove kill file (allow control to resume)
rm /tmp/eidosian_control_kill

# Check status
ls -la /tmp/eidosian_control*

# Force kill by PID
kill -9 $(cat /tmp/eidosian_control.pid)
```

## 🔗 System Integration

- **Eidos MCP Plugin**: Exposed as `computer_control` plugin, providing tools like `computer_type`, `computer_click`, `computer_screenshot`.
- **Agent Forge**: Agents can request "hands-on" tasks via this forge when specialized tooling is required.

## Structure

```
computer_control_forge/
├── README.md           # This file
├── SAFETY.md           # Safety documentation
├── pyproject.toml      # Package config
├── src/
│   └── computer_control_forge/
│       ├── __init__.py
│       ├── control.py      # Main control service
│       ├── keyboard.py     # Keyboard operations
│       ├── mouse.py        # Mouse operations
│       ├── screen.py       # Screen capture
│       ├── safety.py       # Kill switch and safety
│       └── provenance.py   # Action logging
├── tests/
│   ├── test_safety.py
│   └── test_control.py
├── docs/
│   └── safety_manual.md
└── scripts/
    ├── install_deps.sh
    └── kill_all_control.sh
```

## Provenance Format

Every action produces a log entry:

```json
{
  "timestamp": "2026-01-23T08:15:00Z",
  "action": "type_text",
  "parameters": {"text": "Hello"},
  "duration_ms": 150,
  "success": true,
  "kill_switch_checked": true
}
```

## Eidosian Principles Applied

- **Contextual Integrity**: Every feature justified by use case
- **Structure as Control**: Safety architecture prevents misuse
- **Precision as Style**: Minimal, exact operations
- **Transparency**: Full audit trail
- **Non-Destruction**: Read-only screen capture, reversible text

## Status

- [x] Design document
- [ ] Core control.py implementation
- [ ] Safety mechanisms
- [ ] Installation script
- [ ] Tests
- [ ] Documentation

Created: 2026-01-23
Author: Eidos
