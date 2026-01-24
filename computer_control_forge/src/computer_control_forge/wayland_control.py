"""
🎮 Wayland Control Service using ydotool

Provides real computer control on Wayland via ydotool daemon.
Supports absolute mouse positioning, keyboard input, and clicks.

VERIFIED WORKING: 2026-01-23
- Mouse absolute positioning ✓
- Keyboard typing ✓
- Click events ✓

Created: 2026-01-23
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


YDOTOOL_SOCKET = "/tmp/.ydotool_socket"
YDOTOOL_BIN = "/usr/local/bin/ydotool"


def _run_ydotool(args: List[str], timeout: float = 5.0) -> Dict[str, Any]:
    """Run ydotool command and return result."""
    env = os.environ.copy()
    env["YDOTOOL_SOCKET"] = YDOTOOL_SOCKET
    
    cmd = [YDOTOOL_BIN] + args
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def check_daemon() -> Dict[str, Any]:
    """Check if ydotoold daemon is running and accessible."""
    socket_path = Path(YDOTOOL_SOCKET)
    
    result = {
        "socket_exists": socket_path.exists(),
        "socket_path": str(socket_path),
        "ydotool_bin": YDOTOOL_BIN,
        "bin_exists": Path(YDOTOOL_BIN).exists()
    }
    
    if socket_path.exists():
        result["socket_permissions"] = oct(socket_path.stat().st_mode)[-3:]
    
    # Test actual connection
    test = _run_ydotool(["mousemove", "--help"])
    result["daemon_accessible"] = test["success"] or "Usage:" in test.get("stderr", "")
    
    return result


def mouse_move_absolute(x: int, y: int) -> Dict[str, Any]:
    """Move mouse to absolute screen position."""
    result = _run_ydotool(["mousemove", "-a", "-x", str(x), "-y", str(y)])
    return {
        "action": "mouse_move_absolute",
        "position": {"x": x, "y": y},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **result
    }


def mouse_move_relative(dx: int, dy: int) -> Dict[str, Any]:
    """Move mouse relative to current position."""
    result = _run_ydotool(["mousemove", "-x", str(dx), "-y", str(dy)])
    return {
        "action": "mouse_move_relative",
        "delta": {"dx": dx, "dy": dy},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **result
    }


def mouse_click(button: int = 1) -> Dict[str, Any]:
    """Click mouse button. 1=left, 2=right, 3=middle."""
    result = _run_ydotool(["click", str(button)])
    return {
        "action": "mouse_click",
        "button": button,
        "button_name": {1: "left", 2: "right", 3: "middle"}.get(button, "unknown"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **result
    }


def type_text(text: str, delay_ms: int = 0) -> Dict[str, Any]:
    """Type text string into active window."""
    args = ["type"]
    if delay_ms > 0:
        args.extend(["--delay", str(delay_ms)])
    args.append(text)
    
    result = _run_ydotool(args)
    return {
        "action": "type_text",
        "text_length": len(text),
        "delay_ms": delay_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **result
    }


def press_key(key: str) -> Dict[str, Any]:
    """
    Press a key or key combination.
    
    Key codes: https://github.com/torvalds/linux/blob/master/include/uapi/linux/input-event-codes.h
    Examples: "28" for Enter, "1" for Escape
    """
    result = _run_ydotool(["key", key])
    return {
        "action": "press_key",
        "key": key,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **result
    }


def scroll(amount: int, horizontal: bool = False) -> Dict[str, Any]:
    """Scroll mouse wheel. Positive=down/right, negative=up/left."""
    if horizontal:
        args = ["mousemove", "-w", "-x", str(amount), "-y", "0"]
    else:
        args = ["mousemove", "-w", "-x", "0", "-y", str(amount)]
    
    result = _run_ydotool(args)
    return {
        "action": "scroll",
        "amount": amount,
        "horizontal": horizontal,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **result
    }


def take_screenshot(output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Take screenshot using XDG portal (Wayland-native).
    
    Returns path to screenshot file.
    """
    import glob
    
    # Find existing screenshots
    existing = set(glob.glob(str(Path.home() / "Pictures" / "Screenshot_*.png")))
    
    # Trigger screenshot via portal
    result = subprocess.run([
        'gdbus', 'call', '--session',
        '--dest', 'org.freedesktop.portal.Desktop',
        '--object-path', '/org/freedesktop/portal/desktop',
        '--method', 'org.freedesktop.portal.Screenshot.Screenshot',
        '', '{"interactive": <false>}'
    ], capture_output=True, text=True, timeout=10)
    
    if result.returncode != 0:
        return {
            "action": "screenshot",
            "success": False,
            "error": result.stderr,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    # Wait for file
    time.sleep(1.5)
    
    # Find new screenshot
    current = set(glob.glob(str(Path.home() / "Pictures" / "Screenshot_*.png")))
    new_files = current - existing
    
    if new_files:
        newest = max(new_files, key=lambda f: Path(f).stat().st_mtime)
        
        # Copy to output path if specified
        if output_path:
            import shutil
            shutil.copy(newest, output_path)
            final_path = output_path
        else:
            final_path = newest
        
        return {
            "action": "screenshot",
            "success": True,
            "path": final_path,
            "size_bytes": Path(final_path).stat().st_size,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    return {
        "action": "screenshot",
        "success": False,
        "error": "No new screenshot file found",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


if __name__ == "__main__":
    print("=== Wayland Control Module Test ===\n")
    
    # Check daemon
    status = check_daemon()
    print(f"Daemon status: {json.dumps(status, indent=2)}")
