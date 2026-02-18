Loading model... 


▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                    ██    ██
                                    ▀▀    ▀▀

build      : b8083-ae2d3f28a
model      : qwen2.5-1.5b-instruct-q5_k_m.gguf
modalities : text

available commands:
  /exit or Ctrl+C     stop or exit
  /regen              regenerate the last response
  /clear              clear the chat history
  /read               add a text file


> <|im_start|>system
You are an Eidosian Documentation Forge agent. Your task is to generate detailed, accurate, and structurally elegant documentation for the provided source file.
Follow these requirements:
1. File Summary: High-level overview of what the file does.
2. API Documentation: List all functions, classes, and their parameters/methods.
3. Current Status: Accurate assessment of the file's current state based on its content and comments.
4. Potential Future Directions: Eidosian ideas for ... (truncated)

# Current State: computer_control_forge

**Date**: 2026-01-25  
**Status**: Advanced / Production-Ready  
**Version**: 0.1.0  

## 📊 Metrics

| Metric | Value |
|--------|-------|
| **Python Files** | 36 |
| **Lines of Code** | 9,257 |
| **Test Coverage** | ~15% (needs expansion) |
| **Dependencies** | pynput, mss, PIL, tesseract |

## 🏗️ Architecture

The Computer Control Forge provides safe, auditable keyboard/mouse control and screen capture for Eidosian agents. It is the physical interface layer that allows EIDOS to interact with the desktop environment.

### Core Subsystems

```
┌─────────────────────────────────────────────────────────────┐
│                 COMPUTER CONTROL FORGE                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   CONTROL    │  │  PERCEPTION  │  │    SAFETY    │      │
│  │   SERVICE    │  │   SYSTEM     │  │   MONITOR    │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                  │               │
│         ▼                 ▼                  ▼               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Keyboard    │  │  Screen      │  │  Kill Switch │      │
│  │  Mouse       │  │  OCR         │  │  Rate Limit  │      │
│  │  Actions     │  │  Windows     │  │  PID Track   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Key Components

| Component | File | Purpose | Status |
|-----------|------|---------|--------|
| **ControlService** | `control.py` | Main control orchestrator | ✅ Complete |
| **SafetyMonitor** | `safety.py` | Kill switch & safety | ✅ Complete |
| **ScreenPerception** | `perception/screen_state.py` | Screen capture & change detection | ✅ Complete |
| **OCREngine** | `perception/ocr_engine.py` | Text recognition | ✅ Complete |
| **WindowManager** | `perception/window_manager.py` | KWin/window tracking | ✅ Complete |
| **UnifiedPerception** | `perception/unified_perception.py` | Integrated world model | ✅ Complete |
| **AgentLoop** | `agent_loop.py` | Autonomous control loop | ✅ Complete |
| **WaylandControl** | `wayland_control.py` | Wayland-native control | ✅ Complete |
| **LiveController** | `live_controller.py` | Real-time control | ✅ Complete |

## 📁 Directory Structure

```
computer_control_forge/
├── README.md                    # Overview & safety docs
├── INSTALL.md                   # Installation guide
├── CURRENT_STATE.md             # This file
├── src/
│   └── computer_control_forge/
│       ├── __init__.py          # Package init, kill switch API
│       ├── control.py           # ControlService (451 LOC)
│       ├── safety.py            # SafetyMonitor, kill switch (161 LOC)
│       ├── actions.py           # Action definitions
│       ├── agent_loop.py        # Autonomous loop
│       ├── agent_controller.py  # High-level agent interface
│       ├── perception/
│       │   ├── __init__.py
│       │   ├── screen_state.py     # 451 LOC - Frame capture
│       │   ├── ocr_engine.py       # 195 LOC - Tesseract OCR
│       │   ├── window_manager.py   # 395 LOC - KWin DBus
│       │   └── unified_perception.py # 497 LOC - World model
│       ├── *_mouse.py           # Various mouse implementations
│       │   ├── absolute_mouse.py
│       │   ├── advanced_mouse.py
│       │   ├── calibrated_mouse.py
│       │   ├── human_mouse.py   # Human-like movement
│       │   ├── live_mouse.py
│       │   ├── perception_mouse.py
│       │   └── pid_mouse.py     # PID controller
│       ├── wayland_control.py   # Wayland-native via ydotool
│       ├── visual_feedback.py   # Visual debugging
│       ├── feedback_system.py   # Action feedback
│       ├── live_*.py            # Real-time components
│       ├── obs_stream*.py       # Observation streaming
│       ├── cli/
│       │   ├── __init__.py
│       │   └── main.py          # CLI interface
│       ├── daemon/
│       │   └── __init__.py
│       └── automation/
│           └── __init__.py
├── tests/
│   └── test_safety.py           # Safety mechanism tests
├── scripts/
│   └── run_continuation_daemon.py
├── demos/
│   └── demo_patterns.py
├── logs/                        # Action logs
├── screenshots/                 # Captured screens
└── docs/
```

## 🔌 Integrations

| Integration | Purpose | Status |
|-------------|---------|--------|
| **eidos_mcp** | MCP tool exposure | 🔶 Planned |
| **llm_forge** | LLM-driven actions | ✅ Active |
| **agent_forge** | Agent task execution | ✅ Active |
| **glyph_forge** | Screenshot → ASCII | 🔶 Planned |

## 🛡️ Safety Mechanisms

1. **Kill Switch File**: `/tmp/eidosian_control_kill`
   - Create to stop all control immediately
   - Checked before EVERY action
   
2. **PID Tracking**: `/tmp/eidosian_control.pid`
   - Force kill via `kill -9 $(cat /tmp/eidosian_control.pid)`
   
3. **Rate Limiting**: 100ms minimum between actions (configurable)

4. **Full Audit Trail**: JSON logs of every action with timestamps

5. **Dry Run Mode**: Test without executing

## 🐛 Known Issues

1. **Wayland Screenshot**: Portal-based capture has 1s delay
2. **OCR Performance**: Tesseract is slow on large regions
3. **Mouse Implementations**: Multiple approaches exist, need consolidation
4. **Test Coverage**: Only safety tests exist, need more

## 🎯 Key Capabilities

### Demonstrated Functionality (2026-01-23)
- ✅ Calculator automation via visual perception
- ✅ Click detection and verification
- ✅ Type text with human-like timing
- ✅ Cross-session memory persistence test
- ✅ ChatGPT debate (visual perception + typing)

### Perception Features
- Frame-by-frame change detection
- OCR for text recognition
- Window tracking via KWin DBus
- Cursor position monitoring
- Spatial indexing for element lookup

## 📝 Notes

- This forge contains the physical embodiment of EIDOS  
- The 2026-01-23 session proved emergent agency via calculator test  
- Multiple mouse implementations reflect iterative development  
- Wayland support is critical for modern Linux desktops  
- All screenshots in root are from the agency demonstration

---

**Last Verified**: 2026-01-25  
**Maintainer**: EIDOS

[ Prompt: 39.4 t/s | Generation: 4.7 t/s ]

Exiting...