#!/usr/bin/env python3
"""
🎯 Absolute Mouse Controller

Creates a virtual device with EV_ABS support for true absolute positioning.
This bypasses mouse acceleration issues by using absolute coordinates.

Requires: root access (or uinput permissions)

Created: 2026-01-23
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, Any, Optional, Tuple

# Check for root
def check_permissions():
    """Check if we have necessary permissions."""
    if os.geteuid() != 0:
        # Try to check if we can access uinput
        if not os.access('/dev/uinput', os.W_OK):
            return False
    return True


try:
    from evdev import UInput, ecodes as e, AbsInfo
    EVDEV_AVAILABLE = True
except ImportError:
    EVDEV_AVAILABLE = False

try:
    import uinput
    UINPUT_AVAILABLE = True  
except ImportError:
    UINPUT_AVAILABLE = False


class AbsoluteMouseDevice:
    """Virtual mouse device with absolute positioning capability."""
    
    def __init__(self, screen_width: int = 1920, screen_height: int = 1080, name: str = "eidos-abs-mouse"):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.name = name
        self._device = None
        self._backend = None
    
    def _create_evdev_device(self):
        """Create device using python-evdev."""
        capabilities = {
            e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT, e.BTN_MIDDLE],
            e.EV_ABS: [
                (e.ABS_X, AbsInfo(value=0, min=0, max=self.screen_width, fuzz=0, flat=0, resolution=0)),
                (e.ABS_Y, AbsInfo(value=0, min=0, max=self.screen_height, fuzz=0, flat=0, resolution=0)),
            ],
            e.EV_REL: [e.REL_WHEEL, e.REL_HWHEEL],
        }
        self._device = UInput(capabilities, name=self.name)
        self._backend = "evdev"
        return True
    
    def _create_uinput_device(self):
        """Create device using python-uinput."""
        events = [
            uinput.ABS_X + (0, self.screen_width, 0, 0),
            uinput.ABS_Y + (0, self.screen_height, 0, 0),
            uinput.BTN_LEFT,
            uinput.BTN_RIGHT,
            uinput.BTN_MIDDLE,
            uinput.REL_WHEEL,
        ]
        self._device = uinput.Device(events, name=self.name)
        self._backend = "uinput"
        return True
    
    def connect(self) -> Dict[str, Any]:
        """Initialize the virtual device."""
        if not check_permissions():
            return {"success": False, "error": "Need root or uinput write access"}
        
        # Try evdev first (more control), then uinput
        if EVDEV_AVAILABLE:
            try:
                self._create_evdev_device()
                time.sleep(0.5)  # Let device settle
                return {
                    "success": True,
                    "backend": self._backend,
                    "device_name": self.name,
                    "resolution": f"{self.screen_width}x{self.screen_height}"
                }
            except Exception as e:
                pass
        
        if UINPUT_AVAILABLE:
            try:
                self._create_uinput_device()
                time.sleep(0.5)
                return {
                    "success": True,
                    "backend": self._backend,
                    "device_name": self.name,
                    "resolution": f"{self.screen_width}x{self.screen_height}"
                }
            except Exception as e:
                return {"success": False, "error": f"uinput failed: {e}"}
        
        return {"success": False, "error": "No backend available (need evdev or python-uinput)"}
    
    def move_absolute(self, x: int, y: int) -> Dict[str, Any]:
        """Move cursor to absolute position."""
        if not self._device:
            return {"success": False, "error": "Device not connected"}
        
        # Clamp to screen bounds
        x = max(0, min(x, self.screen_width))
        y = max(0, min(y, self.screen_height))
        
        try:
            if self._backend == "evdev":
                self._device.write(e.EV_ABS, e.ABS_X, x)
                self._device.write(e.EV_ABS, e.ABS_Y, y)
                self._device.syn()
            else:
                self._device.emit(uinput.ABS_X, x, syn=False)
                self._device.emit(uinput.ABS_Y, y)
            
            return {"success": True, "position": {"x": x, "y": y}}
        except Exception as err:
            return {"success": False, "error": str(err)}
    
    def click(self, button: str = "left") -> Dict[str, Any]:
        """Click a mouse button."""
        if not self._device:
            return {"success": False, "error": "Device not connected"}
        
        btn_map = {"left": e.BTN_LEFT, "right": e.BTN_RIGHT, "middle": e.BTN_MIDDLE}
        btn = btn_map.get(button, e.BTN_LEFT)
        
        try:
            if self._backend == "evdev":
                self._device.write(e.EV_KEY, btn, 1)
                self._device.syn()
                time.sleep(0.05)
                self._device.write(e.EV_KEY, btn, 0)
                self._device.syn()
            else:
                uinput_btn = getattr(uinput, f"BTN_{button.upper()}")
                self._device.emit(uinput_btn, 1)
                time.sleep(0.05)
                self._device.emit(uinput_btn, 0)
            
            return {"success": True, "button": button}
        except Exception as err:
            return {"success": False, "error": str(err)}
    
    def click_at(self, x: int, y: int, button: str = "left") -> Dict[str, Any]:
        """Move to position and click."""
        move_result = self.move_absolute(x, y)
        if not move_result.get("success"):
            return move_result
        
        time.sleep(0.05)
        return self.click(button)
    
    def disconnect(self):
        """Close the virtual device."""
        if self._device:
            if self._backend == "evdev":
                self._device.close()
            self._device = None
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, *args):
        self.disconnect()


# Singleton instance for easy access
_device: Optional[AbsoluteMouseDevice] = None


def get_absolute_mouse(width: int = 1920, height: int = 1080) -> AbsoluteMouseDevice:
    """Get or create the absolute mouse device."""
    global _device
    if _device is None:
        _device = AbsoluteMouseDevice(width, height)
    return _device


if __name__ == "__main__":
    print("Testing Absolute Mouse Device")
    print("=" * 40)
    
    if not check_permissions():
        print("ERROR: Need root or uinput access")
        print("Run with: sudo python absolute_mouse.py")
        sys.exit(1)
    
    mouse = AbsoluteMouseDevice(1920, 1080)
    result = mouse.connect()
    print(f"Connect result: {result}")
    
    if result.get("success"):
        print("\nMoving to center (960, 540)...")
        print(mouse.move_absolute(960, 540))
        time.sleep(1)
        
        print("\nMoving to corners...")
        for pos in [(100, 100), (1820, 100), (1820, 980), (100, 980), (960, 540)]:
            print(f"  Moving to {pos}")
            mouse.move_absolute(*pos)
            time.sleep(0.5)
        
        mouse.disconnect()
        print("\nDone!")
