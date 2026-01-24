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


class ControlService:
    """
    Safe computer control service with kill switch protection.
    
    Usage:
        control = ControlService()
        control.start()
        
        try:
            control.type_text("Hello!")
            control.click(100, 200)
            img = control.capture_screen()
        finally:
            control.stop()
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
        
        # Lazy-loaded backends
        self._keyboard = None
        self._mouse = None
        self._screen = None
    
    def _on_kill(self) -> None:
        """Called when kill switch is engaged."""
        self._active = False
        self._save_session_log()
    
    def _log_action(self, log: ActionLog) -> None:
        """Log an action."""
        self._actions.append(log)
    
    def _save_session_log(self) -> Path:
        """Save all actions from this session."""
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
        """Enforce minimum time between actions."""
        elapsed = time.time() - self._last_action_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_action_time = time.time()
    
    def _check_safety(self) -> None:
        """Check safety conditions before action."""
        if not self._active:
            raise RuntimeError("ControlService not started")
        self._safety.check()  # Raises KillSwitchEngaged if active
    
    def start(self) -> None:
        """Start the control service."""
        if self._active:
            return
        
        self._safety.start()
        self._active = True
        
        self._log_action(ActionLog(
            action="start",
            parameters={"session_id": self._session_id, "dry_run": self.dry_run},
        ))
    
    def stop(self) -> None:
        """Stop the control service and save logs."""
        if not self._active:
            return
        
        self._active = False
        self._safety.stop()
        
        self._log_action(ActionLog(
            action="stop",
            parameters={"action_count": len(self._actions)},
        ))
        
        self._save_session_log()
    
    def type_text(self, text: str, interval_ms: int = 50) -> None:
        """
        Type text using keyboard.
        
        Args:
            text: Text to type
            interval_ms: Delay between keystrokes in milliseconds
        """
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
        """
        Press a single key.
        
        Args:
            key: Key to press (e.g., "enter", "tab", "a", "ctrl+c")
        """
        self._check_safety()
        self._enforce_rate_limit()
        
        start = time.time()
        log = ActionLog(action="press_key", parameters={"key": key})
        
        try:
            if not self.dry_run:
                self._ensure_keyboard()
                # Handle key combinations
                if "+" in key:
                    parts = key.split("+")
                    # Press modifiers, then key, then release
                    # This is simplified - full impl would use pynput properly
                    pass
                else:
                    self._keyboard.press(key)
                    self._keyboard.release(key)
            
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
        """
        Click at screen position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            button: "left", "right", or "middle"
        """
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
        """
        Move mouse to position.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
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
        """
        Capture screenshot.
        
        Args:
            region: Optional (x, y, width, height) to capture specific region
            
        Returns:
            PNG image bytes, or None in dry_run mode
        """
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
        """Lazy-load keyboard controller."""
        if self._keyboard is None:
            try:
                from pynput.keyboard import Controller
                self._keyboard = Controller()
            except ImportError:
                raise RuntimeError("pynput not installed. Run: pip install pynput")
    
    def _ensure_mouse(self) -> None:
        """Lazy-load mouse controller."""
        if self._mouse is None:
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
                raise RuntimeError("pynput not installed. Run: pip install pynput")
    
    def _ensure_screen(self) -> None:
        """Lazy-load screen capture."""
        if self._screen is None:
            # Try Wayland portal first, then fall back to mss
            import os
            session_type = os.environ.get('XDG_SESSION_TYPE', '')
            
            if session_type == 'wayland':
                # Use portal for Wayland
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


class WaylandScreenCapture:
    """Screen capture using XDG Portal for Wayland."""
    
    def __init__(self):
        self._screenshot_dir = Path("/home/lloyd/eidosian_forge/computer_control_forge/screenshots")
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
    
    def capture(self, region=None) -> Optional[bytes]:
        """
        Capture screenshot using XDG portal.
        Note: Region not supported on Wayland portal.
        """
        import subprocess
        import glob
        import time
        from datetime import datetime
        
        # Find newest screenshot before we take one
        existing = set(glob.glob(str(Path.home() / "Pictures" / "Screenshot_*.png")))
        
        # Use portal via gdbus
        result = subprocess.run([
            'gdbus', 'call', '--session',
            '--dest', 'org.freedesktop.portal.Desktop',
            '--object-path', '/org/freedesktop/portal/desktop',
            '--method', 'org.freedesktop.portal.Screenshot.Screenshot',
            '', '{"interactive": <false>}'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            raise RuntimeError(f"Portal screenshot failed: {result.stderr}")
        
        # Wait for screenshot file to appear
        time.sleep(1)
        
        # Find new screenshot
        current = set(glob.glob(str(Path.home() / "Pictures" / "Screenshot_*.png")))
        new_files = current - existing
        
        if new_files:
            # Get the newest one
            newest = max(new_files, key=lambda f: Path(f).stat().st_mtime)
            
            # Read and return
            with open(newest, 'rb') as f:
                return f.read()
        
        return None
