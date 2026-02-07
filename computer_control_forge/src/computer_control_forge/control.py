"""
🎮 Control Service

Main control service combining keyboard, mouse, and screen operations.
All operations are guarded by safety checks.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .safety import SafetyMonitor, KillSwitchEngaged, is_kill_switch_active


LOG_DIR = Path("/home/lloyd/eidosian_forge/computer_control_forge/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ActionLog:
    """Log entry for a control action."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    action: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    kill_switch_checked: bool = True


class WaylandKeyboard:
    """Wrapper for wayland_control keyboard functions."""
    def type(self, text):
        from . import wayland_control
        wayland_control.type_text(text)
        
    def press(self, key):
        from . import wayland_control
        if hasattr(key, 'name'): 
            k = key.name
        elif hasattr(key, 'char'):
            k = key.char
        else:
            k = str(key)
        
        # Mapping common keys to Linux input event names/codes
        # ydotool accepts KEY_ENTER, 28, etc.
        mapping = {
            "enter": "KEY_ENTER",
            "return": "KEY_ENTER",
            "tab": "KEY_TAB",
            "space": "KEY_SPACE",
            "backspace": "KEY_BACKSPACE",
            "esc": "KEY_ESC",
            "escape": "KEY_ESC",
            "ctrl": "KEY_LEFTCTRL",
            "alt": "KEY_LEFTALT",
            "shift": "KEY_LEFTSHIFT",
            "up": "KEY_UP",
            "down": "KEY_DOWN",
            "left": "KEY_LEFT",
            "right": "KEY_RIGHT",
            "page_up": "KEY_PAGEUP",
            "page_down": "KEY_PAGEDOWN",
            "home": "KEY_HOME",
            "end": "KEY_END",
            "insert": "KEY_INSERT",
            "delete": "KEY_DELETE",
            "f1": "KEY_F1", "f2": "KEY_F2", "f3": "KEY_F3", "f4": "KEY_F4",
            "f5": "KEY_F5", "f6": "KEY_F6", "f7": "KEY_F7", "f8": "KEY_F8",
            "f9": "KEY_F9", "f10": "KEY_F10", "f11": "KEY_F11", "f12": "KEY_F12"
        }
        
        target = mapping.get(k.lower(), k.upper())
        if not target.startswith("KEY_") and len(target) == 1:
            target = f"KEY_{target}"
            
        wayland_control.press_key(f"{target}:1")

    def release(self, key):
        from . import wayland_control
        if hasattr(key, 'name'): 
            k = key.name
        elif hasattr(key, 'char'):
            k = key.char
        else:
            k = str(key)
            
        mapping = {
            "enter": "KEY_ENTER",
            "return": "KEY_ENTER",
            "tab": "KEY_TAB",
            "space": "KEY_SPACE",
            "backspace": "KEY_BACKSPACE",
            "esc": "KEY_ESC",
            "escape": "KEY_ESC",
            "ctrl": "KEY_LEFTCTRL",
            "alt": "KEY_LEFTALT",
            "shift": "KEY_LEFTSHIFT",
            "up": "KEY_UP",
            "down": "KEY_DOWN",
            "left": "KEY_LEFT",
            "right": "KEY_RIGHT",
            "page_up": "KEY_PAGEUP",
            "page_down": "KEY_PAGEDOWN",
            "home": "KEY_HOME",
            "end": "KEY_END",
            "insert": "KEY_INSERT",
            "delete": "KEY_DELETE",
            "f1": "KEY_F1", "f2": "KEY_F2", "f3": "KEY_F3", "f4": "KEY_F4",
            "f5": "KEY_F5", "f6": "KEY_F6", "f7": "KEY_F7", "f8": "KEY_F8",
            "f9": "KEY_F9", "f10": "KEY_F10", "f11": "KEY_F11", "f12": "KEY_F12"
        }
        
        target = mapping.get(k.lower(), k.upper())
        if not target.startswith("KEY_") and len(target) == 1:
            target = f"KEY_{target}"
            
        wayland_control.press_key(f"{target}:0")


class WaylandMouse:
    """Wrapper for wayland_control mouse functions."""
    def __init__(self):
        self._pos = (0, 0)
    
    @property
    def position(self):
        return self._pos
    
    @position.setter
    def position(self, pos):
        from . import wayland_control
        self._pos = pos
        wayland_control.mouse_move_absolute(pos[0], pos[1])
    
    def click(self, button="left"):
        from . import wayland_control
        # Standard Linux input event codes
        # BTN_LEFT = 0x110 (272)
        # BTN_RIGHT = 0x111 (273)
        # BTN_MIDDLE = 0x112 (274)
        btn_map = {
            "left": "0x110",
            "right": "0x111",
            "middle": "0x112"
        }
        b = btn_map.get(button.lower(), "0x110")
        wayland_control.mouse_click(b)


class WaylandScreenCapture:
    """Screen capture using XDG Portal for Wayland."""
    
    def __init__(self):
        self._screenshot_dir = Path("/home/lloyd/eidosian_forge/computer_control_forge/screenshots")
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
    
    def capture(self, region=None) -> Optional[bytes]:
        import subprocess
        import glob
        import time
        from datetime import datetime
        
        existing = set(glob.glob(str(Path.home() / "Pictures" / "Screenshot_*.png")))
        
        result = subprocess.run([
            'gdbus', 'call', '--session',
            '--dest', 'org.freedesktop.portal.Desktop',
            '--object-path', '/org/freedesktop/portal/desktop',
            '--method', 'org.freedesktop.portal.Screenshot.Screenshot',
            '', '{"interactive": <false>}'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            raise RuntimeError(f"Portal screenshot failed: {result.stderr}")
        
        time.sleep(1)
        
        current = set(glob.glob(str(Path.home() / "Pictures" / "Screenshot_*.png")))
        new_files = current - existing
        
        if new_files:
            newest = max(new_files, key=lambda f: Path(f).stat().st_mtime)
            with open(newest, 'rb') as f:
                return f.read()
        return None


class ControlService:
    """
    Safe computer control service with kill switch protection.
    """
    
    def __init__(
        self,
        rate_limit_ms: int = 100,
        log_dir: Optional[Path] = None,
        dry_run: bool = False,
    ):
        self.rate_limit = rate_limit_ms / 1000.0
        self.log_dir = log_dir or LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        
        self._session_id = str(uuid.uuid4())
        self._safety = SafetyMonitor(on_kill=self._on_kill)
        self._actions: List[ActionLog] = []
        self._last_action_time = 0.0
        self._active = False
        
        self._keyboard = None
        self._mouse = None
        self._screen = None
    
    def _on_kill(self) -> None:
        self._active = False
        self._save_session_log()
    
    def _log_action(self, log: ActionLog) -> None:
        self._actions.append(log)
    
    def _save_session_log(self) -> Path:
        path = self.log_dir / f"session_{self._session_id}.json"
        with path.open("w") as f:
            json.dump(
                {
                    "session_id": self._session_id,
                    "start_time": self._actions[0].timestamp if self._actions else None,
                    "end_time": datetime.now(timezone.utc).isoformat(),
                    "action_count": len(self._actions),
                    "dry_run": self.dry_run,
                    "actions": [
                        {
                            "id": a.id,
                            "timestamp": a.timestamp,
                            "action": a.action,
                            "parameters": a.parameters,
                            "duration_ms": a.duration_ms,
                            "success": a.success,
                            "error": a.error,
                        }
                        for a in self._actions
                    ],
                },
                f,
                indent=2,
            )
        return path
    
    def _enforce_rate_limit(self) -> None:
        elapsed = time.time() - self._last_action_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_action_time = time.time()
    
    def _check_safety(self) -> None:
        if not self._active:
            raise RuntimeError("ControlService not started")
        self._safety.check()
    
    def start(self) -> None:
        if self._active:
            return
        self._safety.start()
        self._active = True
        self._log_action(ActionLog(action="start", parameters={"session_id": self._session_id, "dry_run": self.dry_run}))
    
    def stop(self) -> None:
        if not self._active:
            return
        self._active = False
        self._safety.stop()
        self._log_action(ActionLog(action="stop", parameters={"action_count": len(self._actions)}))
        self._save_session_log()
    
    def type_text(self, text: str, interval_ms: int = 50) -> None:
        self._check_safety()
        self._enforce_rate_limit()
        start = time.time()
        log = ActionLog(action="type_text", parameters={"text": text, "interval_ms": interval_ms})
        try:
            if not self.dry_run:
                self._ensure_keyboard()
                for char in text:
                    self._safety.check()
                    self._keyboard.type(char)
                    time.sleep(interval_ms / 1000.0)
            log.duration_ms = (time.time() - start) * 1000
            log.success = True
        except KillSwitchEngaged:
            log.success = False
            log.error = "Kill switch engaged"
            raise
        except Exception as e:
            log.success = False
            log.error = str(e)
            raise
        finally:
            self._log_action(log)
    
    def press_key(self, key: str) -> None:
        self._check_safety()
        self._enforce_rate_limit()
        start = time.time()
        log = ActionLog(action="press_key", parameters={"key": key})
        try:
            if not self.dry_run:
                self._ensure_keyboard()
                
                # Check if we are using Wayland backend
                if isinstance(self._keyboard, WaylandKeyboard):
                    if "+" in key:
                        parts = key.split("+")
                        for p in parts:
                            self._keyboard.press(p)
                        for p in reversed(parts):
                            self._keyboard.release(p)
                    else:
                        self._keyboard.press(key)
                        self._keyboard.release(key)
                else:
                    # Pynput path
                    if "+" in key:
                        from pynput.keyboard import Key
                        parts = key.split("+")
                        keys = []
                        for part in parts:
                            if hasattr(Key, part):
                                keys.append(getattr(Key, part))
                            else:
                                keys.append(part)
                        for k in keys:
                            self._keyboard.press(k)
                        for k in reversed(keys):
                            self._keyboard.release(k)
                    else:
                        from pynput.keyboard import Key
                        if hasattr(Key, key):
                            k = getattr(Key, key)
                        else:
                            k = key
                        self._keyboard.press(k)
                        self._keyboard.release(k)
            log.duration_ms = (time.time() - start) * 1000
            log.success = True
        except KillSwitchEngaged:
            log.success = False
            log.error = "Kill switch engaged"
            raise
        except Exception as e:
            log.success = False
            log.error = str(e)
            raise
        finally:
            self._log_action(log)
    
    def click(self, x: int, y: int, button: str = "left") -> None:
        self._check_safety()
        self._enforce_rate_limit()
        start = time.time()
        log = ActionLog(action="click", parameters={"x": x, "y": y, "button": button})
        try:
            if not self.dry_run:
                self._ensure_mouse()
                self._mouse.position = (x, y)
                self._mouse.click(button)
            log.duration_ms = (time.time() - start) * 1000
            log.success = True
        except KillSwitchEngaged:
            log.success = False
            log.error = "Kill switch engaged"
            raise
        except Exception as e:
            log.success = False
            log.error = str(e)
            raise
        finally:
            self._log_action(log)
    
    def move_mouse(self, x: int, y: int) -> None:
        self._check_safety()
        self._enforce_rate_limit()
        start = time.time()
        log = ActionLog(action="move_mouse", parameters={"x": x, "y": y})
        try:
            if not self.dry_run:
                self._ensure_mouse()
                self._mouse.position = (x, y)
            log.duration_ms = (time.time() - start) * 1000
            log.success = True
        except Exception as e:
            log.success = False
            log.error = str(e)
            raise
        finally:
            self._log_action(log)
    
    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[bytes]:
        self._check_safety()
        start = time.time()
        log = ActionLog(action="capture_screen", parameters={"region": region})
        try:
            result = None
            if not self.dry_run:
                self._ensure_screen()
                result = self._screen.capture(region)
            log.duration_ms = (time.time() - start) * 1000
            log.success = True
            return result
        except Exception as e:
            log.success = False
            log.error = str(e)
            raise
        finally:
            self._log_action(log)
            
    def _ensure_keyboard(self) -> None:
        if self._keyboard is None:
            import os
            if os.environ.get('XDG_SESSION_TYPE') == 'wayland':
                try:
                    from . import wayland_control
                    if wayland_control.check_daemon()['daemon_accessible']:
                        self._keyboard = WaylandKeyboard()
                        return
                except Exception:
                    pass
            try:
                from pynput.keyboard import Controller
                self._keyboard = Controller()
            except ImportError:
                 if os.environ.get('XDG_SESSION_TYPE') == 'wayland':
                     self._keyboard = WaylandKeyboard()
                 else:
                    raise RuntimeError("pynput not installed")
            except Exception as e:
                 try:
                    from . import wayland_control
                    if wayland_control.check_daemon()['daemon_accessible']:
                        self._keyboard = WaylandKeyboard()
                        return
                 except:
                     pass
                 raise e

    def _ensure_mouse(self) -> None:
        if self._mouse is None:
            import os
            if os.environ.get('XDG_SESSION_TYPE') == 'wayland':
                try:
                    from . import wayland_control
                    if wayland_control.check_daemon()['daemon_accessible']:
                        self._mouse = WaylandMouse()
                        return
                except Exception:
                    pass
            try:
                from pynput.mouse import Controller, Button
                class MouseWrapper:
                    def __init__(self):
                        self._controller = Controller()
                        self._buttons = {"left": Button.left, "right": Button.right, "middle": Button.middle}
                    @property
                    def position(self):
                        return self._controller.position
                    @position.setter
                    def position(self, pos):
                        self._controller.position = pos
                    def click(self, button="left"):
                        self._controller.click(self._buttons.get(button, Button.left))
                self._mouse = MouseWrapper()
            except ImportError:
                if os.environ.get('XDG_SESSION_TYPE') == 'wayland':
                     self._mouse = WaylandMouse()
                else:
                    raise RuntimeError("pynput not installed")
            except Exception as e:
                 try:
                    from . import wayland_control
                    if wayland_control.check_daemon()['daemon_accessible']:
                        self._mouse = WaylandMouse()
                        return
                 except:
                     pass
                 raise e

    def _ensure_screen(self) -> None:
        if self._screen is None:
            import os
            session_type = os.environ.get('XDG_SESSION_TYPE', '')
            if session_type == 'wayland':
                self._screen = WaylandScreenCapture()
            else:
                try:
                    import mss
                    from PIL import Image
                    import io
                    class ScreenCapture:
                        def __init__(self):
                            self._sct = mss.mss()
                        def capture(self, region=None):
                            if region:
                                monitor = {"left": region[0], "top": region[1], "width": region[2], "height": region[3]}
                            else:
                                monitor = self._sct.monitors[0]
                            sct_img = self._sct.grab(monitor)
                            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                            buffer = io.BytesIO()
                            img.save(buffer, format="PNG")
                            return buffer.getvalue()
                    self._screen = ScreenCapture()
                except ImportError as e:
                    raise RuntimeError(f"Screen capture dependencies not installed: {e}")

    def __enter__(self) -> "ControlService":
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()