"""
🛡️ Eidosian Safety Module

Provides mandatory kill switch mechanisms and operational safeguards for 
computer control actuators. This module ensures that autonomous actions 
remain under deterministic human or systemic control.
"""

from __future__ import annotations

import os
import signal
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from eidosian_core import eidosian

KILL_FILE = Path("/tmp/eidosian_control_kill")
PID_FILE = Path("/tmp/eidosian_control.pid")


class KillSwitchEngaged(Exception):
    """Raised when the systemic or manual kill switch has been activated."""
    pass


class SafetyMonitor:
    """
    Continuous background monitor for safety constraints and kill-switch state.
    
    Provides both polling (`check()`) and threaded callback mechanisms to ensure 
    that control operations are terminated immediately upon safety breach.
    """
    
    def __init__(
        self,
        on_kill: Optional[Callable[[], None]] = None,
        check_interval_ms: int = 100,
    ):
        """
        Initialize the Safety Monitor.
        
        Args:
            on_kill (Optional[Callable]): Callback triggered when kill switch is detected.
            check_interval_ms (int): Polling frequency for the background thread.
        """
        self.on_kill = on_kill
        self.check_interval = check_interval_ms / 1000.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._pid_written = False
    
    @eidosian()
    def start(self) -> None:
        """
        Activates the safety monitor and registers the current process PID.
        
        Raises:
            KillSwitchEngaged: If the kill-switch is already active.
        """
        if self._running:
            return
        
        # Check kill switch before starting
        if KILL_FILE.exists():
            raise KillSwitchEngaged("Kill switch is engaged. Remove /tmp/eidosian_control_kill to proceed.")
        
        # Write PID file
        PID_FILE.write_text(str(os.getpid()))
        self._pid_written = True
        
        # Start monitoring thread
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    @eidosian()
    def stop(self) -> None:
        """Deactivates the safety monitor and cleans up process registry."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        
        # Clean up PID file
        if self._pid_written and PID_FILE.exists():
            try:
                PID_FILE.unlink()
            except Exception:
                pass
            self._pid_written = False
    
    @eidosian()
    def check(self) -> None:
        """
        Synchronous check for kill-switch status.
        
        Should be called before every motor or sensory action.
        
        Raises:
            KillSwitchEngaged: If the kill-switch is active.
        """
        if KILL_FILE.exists():
            self._handle_kill()
            raise KillSwitchEngaged("Kill switch engaged")
    
    def _monitor_loop(self) -> None:
        """Background loop checking for kill switch."""
        while self._running:
            if KILL_FILE.exists():
                self._handle_kill()
                break
            time.sleep(self.check_interval)
    
    def _handle_kill(self) -> None:
        """Internal handler for kill-switch activation."""
        self._running = False
        
        # Clean up PID file
        if self._pid_written and PID_FILE.exists():
            try:
                PID_FILE.unlink()
            except Exception:
                pass
            self._pid_written = False
        
        # Call handler
        if self.on_kill:
            try:
                self.on_kill()
            except Exception:
                pass


@eidosian()
def engage_kill_switch() -> None:
    """Manually engages the global kill switch."""
    KILL_FILE.write_text("KILL")


@eidosian()
def disengage_kill_switch() -> None:
    """Manually disengages the global kill switch."""
    if KILL_FILE.exists():
        KILL_FILE.unlink()


@eidosian()
def is_kill_switch_active() -> bool:
    """Returns True if the kill switch is currently active."""
    return KILL_FILE.exists()


@eidosian()
def get_control_pid() -> Optional[int]:
    """Retrieves the PID of the currently active control service."""
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text().strip())
    except Exception:
        return None


@eidosian()
def force_stop_control() -> bool:
    """Forcefully terminates the active control process via SIGTERM."""
    pid = get_control_pid()
    if pid is None:
        return False
    
    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except ProcessLookupError:
        # Process already dead, clean up PID file
        if PID_FILE.exists():
            PID_FILE.unlink()
        return False
    except Exception:
        return False
