from eidosian_core import eidosian
"""
Computer Control Plugin v5.0.0

Provides verified mouse and keyboard control for KDE Wayland.
Now with multi-modal live feedback system.

Author: Eidos
Version: 5.0.0
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add computer_control_forge to path
sys.path.insert(0, '/home/lloyd/eidosian_forge/computer_control_forge/src')

PLUGIN_MANIFEST = {
    "name": "computer_control",
    "version": "5.0.0",
    "description": "Verified mouse and keyboard control with live feedback for KDE Wayland",
    "author": "Eidos",
    "tools": [
        "control_get_cursor_position",
        "control_move_mouse_to",
        "control_move_mouse_relative",
        "control_mouse_click",
        "control_type_text",
        "control_press_key",
        "control_screenshot",
        "control_ocr_screen",
        "control_system_status",
        "control_live_move",
        "control_live_click",
        "control_live_type",
        "control_get_live_events",
        "control_start_monitoring",
        "control_get_state"
    ]
}

# Global live controller instance
_live_controller = None

def _get_live_controller():
    """Get or create live controller singleton."""
    global _live_controller
    if _live_controller is None:
        from computer_control_forge.live_controller import LiveController, ControllerConfig
        config = ControllerConfig(
            enable_stdout=False,  # Don't spam stdout in MCP context
            enable_file=True,
            file_path="/tmp/eidos_live.jsonl"
        )
        _live_controller = LiveController(config)
    return _live_controller

# Ensure ydotool socket
os.environ.setdefault("YDOTOOL_SOCKET", "/tmp/.ydotool_socket")


def _get_calibrated_mouse():
    """Get calibrated mouse controller."""
    from computer_control_forge.calibrated_mouse import CalibratedMouse
    return CalibratedMouse()


def _get_cursor_position():
    """Get cursor position using KWin scripting."""
    from computer_control_forge.cursor_position import get_cursor_position
    return get_cursor_position()


@eidosian()
def control_get_cursor_position() -> Dict[str, Any]:
    """
    Get current cursor position on screen.
    
    Returns:
        Dictionary with x, y coordinates and timestamp
    """
    try:
        x, y = _get_cursor_position()
        return {
            "success": True,
            "x": x,
            "y": y,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@eidosian()
def control_move_mouse_to(x: int, y: int, tolerance: int = 10) -> Dict[str, Any]:
    """
    Move mouse cursor to absolute screen position.
    
    Uses calibrated movement with position feedback to ensure accuracy
    despite mouse acceleration.
    
    Args:
        x: Target X coordinate
        y: Target Y coordinate
        tolerance: Acceptable error in pixels (default: 10)
        
    Returns:
        Dictionary with movement result including actual position and error
    """
    try:
        mouse = _get_calibrated_mouse()
        result = mouse.move_to(x, y, tolerance=tolerance)
        return {
            "success": result.success,
            "target": {"x": result.target[0], "y": result.target[1]},
            "actual": {"x": result.actual[0], "y": result.actual[1]},
            "error": {"x": result.error[0], "y": result.error[1]},
            "error_distance": result.error_distance,
            "attempts": result.attempts,
            "duration_ms": result.duration_ms,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@eidosian()
def control_move_mouse_relative(dx: int, dy: int) -> Dict[str, Any]:
    """
    Move mouse cursor by relative amount.
    
    Args:
        dx: Horizontal movement (positive = right)
        dy: Vertical movement (positive = down)
        
    Returns:
        Dictionary with new position
    """
    try:
        result = subprocess.run(
            ["ydotool", "mousemove", "-x", str(dx), "-y", str(dy)],
            capture_output=True, timeout=5
        )
        if result.returncode == 0:
            # Get new position
            x, y = _get_cursor_position()
            return {
                "success": True,
                "moved": {"dx": dx, "dy": dy},
                "new_position": {"x": x, "y": y},
                "timestamp": datetime.now().isoformat()
            }
        return {"success": False, "error": "ydotool command failed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@eidosian()
def control_mouse_click(button: str = "left", x: Optional[int] = None, y: Optional[int] = None) -> Dict[str, Any]:
    """
    Click mouse button, optionally at a specific position.
    
    Args:
        button: "left", "right", or "middle"
        x: Optional X coordinate to click at
        y: Optional Y coordinate to click at
        
    Returns:
        Dictionary with click result and position
    """
    try:
        if x is not None and y is not None:
            mouse = _get_calibrated_mouse()
            move_result = mouse.move_to(x, y)
            if not move_result.success:
                return {
                    "success": False,
                    "error": f"Failed to move to position: error={move_result.error}"
                }
            position = move_result.actual
        else:
            position = _get_cursor_position()
        
        button_codes = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}
        code = button_codes.get(button, "0xC0")
        
        result = subprocess.run(
            ["ydotool", "click", code],
            capture_output=True, timeout=5
        )
        
        return {
            "success": result.returncode == 0,
            "button": button,
            "position": {"x": position[0], "y": position[1]},
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@eidosian()
def control_type_text(text: str, delay_ms: int = 0) -> Dict[str, Any]:
    """
    Type text using keyboard.
    
    Args:
        text: Text to type
        delay_ms: Delay between keystrokes in milliseconds
        
    Returns:
        Dictionary with typing result
    """
    try:
        cmd = ["ydotool", "type"]
        if delay_ms > 0:
            cmd.extend(["--key-delay", str(delay_ms)])
        cmd.append(text)
        
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        
        return {
            "success": result.returncode == 0,
            "text_length": len(text),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@eidosian()
def control_press_key(key: str) -> Dict[str, Any]:
    """
    Press a special key or key combination.
    
    Args:
        key: Key code (e.g., "enter", "tab", "ctrl+c", "alt+f4")
        
    Returns:
        Dictionary with result
    """
    try:
        # Map common key names to keycodes
        key_map = {
            "enter": "28:1 28:0",
            "tab": "15:1 15:0",
            "escape": "1:1 1:0",
            "backspace": "14:1 14:0",
            "space": "57:1 57:0",
            "up": "103:1 103:0",
            "down": "108:1 108:0",
            "left": "105:1 105:0",
            "right": "106:1 106:0",
        }
        
        key_code = key_map.get(key.lower())
        if not key_code:
            return {"success": False, "error": f"Unknown key: {key}"}
        
        result = subprocess.run(
            ["ydotool", "key", key_code],
            capture_output=True, timeout=5
        )
        
        return {
            "success": result.returncode == 0,
            "key": key,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@eidosian()
def control_screenshot(output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Take a screenshot.
    
    Args:
        output_path: Optional path to save screenshot (default: temp file)
        
    Returns:
        Dictionary with screenshot path and dimensions
    """
    try:
        if output_path is None:
            output_path = f"/tmp/eidos_screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        result = subprocess.run(
            ["spectacle", "-b", "-n", "-o", output_path],
            capture_output=True, timeout=10
        )
        
        if result.returncode == 0 and os.path.exists(output_path):
            # Get image dimensions
            from PIL import Image
            with Image.open(output_path) as img:
                width, height = img.size
            
            return {
                "success": True,
                "path": output_path,
                "width": width,
                "height": height,
                "size_bytes": os.path.getsize(output_path),
                "timestamp": datetime.now().isoformat()
            }
        return {"success": False, "error": "Screenshot failed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@eidosian()
def control_ocr_screen(region: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Perform OCR on screen or region.
    
    Args:
        region: Optional dict with x, y, width, height for partial OCR
        
    Returns:
        Dictionary with extracted text
    """
    try:
        import pytesseract
        from PIL import Image
        
        # Take screenshot first
        ss_result = control_screenshot()
        if not ss_result.get("success"):
            return ss_result
        
        img = Image.open(ss_result["path"])
        
        if region:
            x = region.get("x", 0)
            y = region.get("y", 0)
            w = region.get("width", img.width - x)
            h = region.get("height", img.height - y)
            img = img.crop((x, y, x + w, y + h))
        
        text = pytesseract.image_to_string(img)
        
        return {
            "success": True,
            "text": text.strip(),
            "text_length": len(text.strip()),
            "region": region,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@eidosian()
def control_system_status() -> Dict[str, Any]:
    """
    Get computer control system status.
    
    Returns:
        Dictionary with status of all components
    """
    status = {
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    # Check ydotoold
    try:
        result = subprocess.run(["pgrep", "-x", "ydotoold"], capture_output=True)
        status["components"]["ydotoold"] = {
            "running": result.returncode == 0,
            "socket": os.path.exists("/tmp/.ydotool_socket")
        }
    except:
        status["components"]["ydotoold"] = {"running": False, "error": "check failed"}
    
    # Check cursor position capability
    try:
        x, y = _get_cursor_position()
        status["components"]["cursor_tracking"] = {
            "working": True,
            "position": {"x": x, "y": y}
        }
    except Exception as e:
        status["components"]["cursor_tracking"] = {"working": False, "error": str(e)}
    
    # Check spectacle
    try:
        result = subprocess.run(["which", "spectacle"], capture_output=True)
        status["components"]["screenshot"] = {"available": result.returncode == 0}
    except:
        status["components"]["screenshot"] = {"available": False}
    
    # Check tesseract
    try:
        result = subprocess.run(["tesseract", "--version"], capture_output=True)
        status["components"]["ocr"] = {"available": result.returncode == 0}
    except:
        status["components"]["ocr"] = {"available": False}
    
    return status


# ============================================================================
# LIVE FEEDBACK TOOLS (v5.0.0)
# ============================================================================

@eidosian()
def control_live_move(x: int, y: int) -> Dict[str, Any]:
    """
    Move mouse to position with full live feedback.
    
    Returns detailed movement data including path, timing, and all events.
    
    Args:
        x: Target X coordinate
        y: Target Y coordinate
        
    Returns:
        Dictionary with movement result and accumulated events
    """
    try:
        ctrl = _get_live_controller()
        result = ctrl.move_to(x, y)
        events = ctrl.get_mcp_events(clear=True)
        return {
            "success": result["success"],
            "result": result,
            "events": events["events"],
            "event_count": events["event_count"]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@eidosian()
def control_live_click(x: int, y: int, button: str = "left") -> Dict[str, Any]:
    """
    Move to position and click with full live feedback.
    
    Args:
        x: Target X coordinate
        y: Target Y coordinate
        button: Button to click ("left", "right", "middle")
        
    Returns:
        Dictionary with click result and accumulated events
    """
    try:
        ctrl = _get_live_controller()
        result = ctrl.click_at(x, y, button)
        events = ctrl.get_mcp_events(clear=True)
        return {
            "success": result["success"],
            "result": result,
            "events": events["events"],
            "event_count": events["event_count"]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@eidosian()
def control_live_type(text: str, wpm: int = 180) -> Dict[str, Any]:
    """
    Type text with live progress feedback.
    
    Args:
        text: Text to type
        wpm: Words per minute (default 180)
        
    Returns:
        Dictionary with typing result and progress events
    """
    try:
        ctrl = _get_live_controller()
        result = ctrl.type_text(text, wpm)
        events = ctrl.get_mcp_events(clear=True)
        return {
            "success": result["success"],
            "result": result,
            "events": events["events"],
            "event_count": events["event_count"]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@eidosian()
def control_get_live_events(clear: bool = True) -> Dict[str, Any]:
    """
    Get accumulated live feedback events.
    
    Args:
        clear: Whether to clear events after reading (default True)
        
    Returns:
        Dictionary with list of events since last read
    """
    try:
        ctrl = _get_live_controller()
        return ctrl.get_mcp_events(clear=clear)
    except Exception as e:
        return {"success": False, "error": str(e), "events": []}


@eidosian()
def control_start_monitoring(duration_seconds: float = 10.0) -> Dict[str, Any]:
    """
    Start background monitoring and return after duration.
    
    Monitors cursor position, window focus, and emits events.
    
    Args:
        duration_seconds: How long to monitor (default 10s, max 60s)
        
    Returns:
        Dictionary with all events captured during monitoring
    """
    import time
    
    try:
        duration = min(60.0, max(1.0, duration_seconds))
        ctrl = _get_live_controller()
        
        ctrl.start_monitoring()
        time.sleep(duration)
        ctrl.stop_monitoring()
        
        events = ctrl.get_mcp_events(clear=True)
        return {
            "success": True,
            "duration": duration,
            "events": events["events"],
            "event_count": events["event_count"]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@eidosian()
def control_get_state() -> Dict[str, Any]:
    """
    Get current controller state.
    
    Returns cursor position, active window, and feedback channel stats.
    """
    try:
        ctrl = _get_live_controller()
        return ctrl.get_state()
    except Exception as e:
        return {"success": False, "error": str(e)}


# Tool registration
TOOLS = {
    "control_get_cursor_position": control_get_cursor_position,
    "control_move_mouse_to": control_move_mouse_to,
    "control_move_mouse_relative": control_move_mouse_relative,
    "control_mouse_click": control_mouse_click,
    "control_type_text": control_type_text,
    "control_press_key": control_press_key,
    "control_screenshot": control_screenshot,
    "control_ocr_screen": control_ocr_screen,
    "control_system_status": control_system_status,
    # Live feedback tools (v5.0.0)
    "control_live_move": control_live_move,
    "control_live_click": control_live_click,
    "control_live_type": control_live_type,
    "control_get_live_events": control_get_live_events,
    "control_start_monitoring": control_start_monitoring,
    "control_get_state": control_get_state,
}


@eidosian()
def get_tools():
    """Return all tools provided by this plugin."""
    return TOOLS


@eidosian()
def get_manifest():
    """Return plugin manifest."""
    return PLUGIN_MANIFEST


if __name__ == "__main__":
    print("Computer Control Plugin v4.0.0")
    print("=" * 40)
    print("\nSystem Status:")
    status = control_system_status()
    print(json.dumps(status, indent=2))
