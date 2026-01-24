#!/usr/bin/env python3
"""
Action System for Eidos

Provides clean interfaces for:
- Mouse movement (absolute positioning)
- Mouse clicks
- Keyboard typing
- Special keys
"""

import subprocess
import time
import os

os.environ.setdefault('YDOTOOL_SOCKET', '/tmp/.ydotool_socket')

class Actions:
    """Mouse and keyboard actions."""
    
    def __init__(self, socket_path: str = '/tmp/.ydotool_socket'):
        self.socket = socket_path
        os.environ['YDOTOOL_SOCKET'] = socket_path
    
    def move_to(self, x: int, y: int) -> bool:
        """Move cursor to absolute position."""
        try:
            result = subprocess.run(
                ['ydotool', 'mousemove', '-a', '-x', str(x), '-y', str(y)],
                capture_output=True, timeout=5
            )
            time.sleep(0.1)
            return result.returncode == 0
        except Exception as e:
            print(f"Move failed: {e}")
            return False
    
    def click(self, button: str = 'left') -> bool:
        """Click mouse button."""
        button_codes = {'left': '0xC0', 'right': '0xC1', 'middle': '0xC2'}
        code = button_codes.get(button, '0xC0')
        try:
            result = subprocess.run(
                ['ydotool', 'click', code],
                capture_output=True, timeout=5
            )
            time.sleep(0.1)
            return result.returncode == 0
        except Exception as e:
            print(f"Click failed: {e}")
            return False
    
    def click_at(self, x: int, y: int, button: str = 'left') -> bool:
        """Move to position and click."""
        if self.move_to(x, y):
            time.sleep(0.15)
            return self.click(button)
        return False
    
    def type_text(self, text: str, delay_ms: int = 5) -> bool:
        """Type text."""
        try:
            result = subprocess.run(
                ['ydotool', 'type', '--key-delay', str(delay_ms), text],
                capture_output=True, timeout=30
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Type failed: {e}")
            return False
    
    def press_key(self, key: str) -> bool:
        """Press a special key."""
        # Key codes for common keys
        key_codes = {
            'enter': '28:1 28:0',
            'tab': '15:1 15:0',
            'escape': '1:1 1:0',
            'backspace': '14:1 14:0',
            'delete': '111:1 111:0',
            'up': '103:1 103:0',
            'down': '108:1 108:0',
            'left': '105:1 105:0',
            'right': '106:1 106:0',
            'home': '102:1 102:0',
            'end': '107:1 107:0',
            'ctrl+a': '29:1 30:1 30:0 29:0',
            'ctrl+c': '29:1 46:1 46:0 29:0',
            'ctrl+v': '29:1 47:1 47:0 29:0',
        }
        code = key_codes.get(key.lower())
        if not code:
            print(f"Unknown key: {key}")
            return False
        try:
            result = subprocess.run(
                ['ydotool', 'key'] + code.split(),
                capture_output=True, timeout=5
            )
            return result.returncode == 0
        except Exception as e:
            print(f"Key press failed: {e}")
            return False
    
    def double_click(self, x: int = None, y: int = None) -> bool:
        """Double click at position (or current if not specified)."""
        if x is not None and y is not None:
            self.move_to(x, y)
            time.sleep(0.1)
        self.click()
        time.sleep(0.05)
        return self.click()


if __name__ == "__main__":
    print("Testing Actions...")
    actions = Actions()
    
    # Test move
    print("Moving to (500, 500)...")
    actions.move_to(500, 500)
    time.sleep(0.5)
    
    print("Moving to (300, 300)...")
    actions.move_to(300, 300)
    
    print("Actions system ready.")
