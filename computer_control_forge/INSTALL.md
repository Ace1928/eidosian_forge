# 🖥️ Computer Control Forge Installation

Mouse, keyboard, and screen control for automation and accessibility.

## Prerequisites

- Python 3.10+
- X11 (Linux) or native APIs (Windows/macOS)
- Optional: Virtual display for headless operation

## Quick Install

```bash
pip install -e ./computer_control_forge

# Install system dependencies (Linux)
sudo apt install python3-xlib scrot

# Verify
python -c "from computer_control_forge import MouseController; print('✓ Ready')"
```

## CLI Usage

```bash
# Move mouse
computer-control mouse move 100 200

# Click
computer-control mouse click

# Type text
computer-control keyboard type "Hello, World!"

# Take screenshot
computer-control screen capture screenshot.png

# Help
computer-control --help
```

## Python API

```python
from computer_control_forge import MouseController, KeyboardController, Screen

# Mouse control
mouse = MouseController()
mouse.move(100, 200)
mouse.click()

# Keyboard control
keyboard = KeyboardController()
keyboard.type("Hello!")

# Screen capture
screen = Screen()
screenshot = screen.capture()
```

## Dependencies

- `pyautogui` - Cross-platform GUI automation
- `Pillow` - Image processing
- `eidosian_core` - Universal decorators and logging

