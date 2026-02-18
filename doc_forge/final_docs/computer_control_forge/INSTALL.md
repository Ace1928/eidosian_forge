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

# 🖥️ Computer Control Forge Installation

Mouse, keyboard, and screen control for automation and accessibility.

## Prerequisites

To use this software, ensure you have the following prerequisites installed on your system:

- Python 3.10+
- X11 (Linux) or native APIs (Windows/macOS)
- Optional: Virtual display for headless operation

## Quick Install

To get started quickly, follow these steps:

```bash
pip install -e ./computer_control_forge

# Verify the installation
python -c "from computer_control_forge import MouseController; print('✓ Ready')"
```

## CLI Usage

For command-line interface usage, use the following commands:

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

For more advanced functionality, use the following Python API:

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

This software relies on several dependencies:

- `pyautogui` - Cross-platform GUI automation library
- `Pillow` - Image processing library
- `eidosian_core` - Universal decorators and logging package

These libraries are installed via pip, so you can install them using the following command:

```bash
pip install pyautogui Pillow eidosian_core
```

## Potential Future Directions

1. **Enhanced User Interface:** Develop a more intuitive user interface for easier navigation.
2. **Integration with Other Tools:** Integrate Computer Control Forge with popular automation tools like Selenium or PyAutoGUI.
3. **Virtual Display Support:** Implement virtual display support on Windows/macOS to simplify testing in environments without physical hardware.
4. **More Robust Error Handling:** Improve error handling and provide more detailed logging for better diagnostics.

By implementing these features, we can further enhance the functionality of Computer Control Forge, making it a powerful tool for automation and accessibility across various platforms and devices.

[ Prompt: 34.0 t/s | Generation: 3.4 t/s ]

Exiting...