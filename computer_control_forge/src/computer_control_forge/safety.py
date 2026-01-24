"""
🛡️ Safety Module

Kill switch and safety mechanisms for computer control.
"""

from __future__ import annotations

import os
import signal
import threading
import time
from pathlib import Path
from typing import Callable, Optional

KILL_FILE = Path("/tmp/eidosian_control_kill")
PID_FILE = Path("/tmp/eidosian_control.pid")


class KillSwitchEngaged(Exception):
    """Raised when kill switch is active."""
    pass


class SafetyMonitor:
    """
    Monitors safety conditions and enforces kill switch.
    
    Usage:
        monitor = SafetyMonitor(on_kill=lambda: print("Killed!"))
        monitor.start()
        
        # Check before each action
        monitor.check()  # Raises KillSwitchEngaged if kill file exists
        
        monitor.stop()
    """
    
    def __init__(
        self,
        on_kill: Optional[Callable[[], None]] = None,
        check_interval_ms: int = 100,
    ):
        self.on_kill = on_kill
        self.check_interval = check_interval_ms / 1000.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._pid_written = False
    
    def start(self) -> None:
        """Start the safety monitor."""
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
    
    def stop(self) -> None:
        """Stop the safety monitor."""
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
    
    def check(self) -> None:
        """
        Check if kill switch is engaged.
        Raises KillSwitchEngaged if active.
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
        """Handle kill switch activation."""
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


def engage_kill_switch() -> None:
    """Engage the kill switch (creates kill file)."""
    KILL_FILE.write_text("KILL")


def disengage_kill_switch() -> None:
    """Disengage the kill switch (removes kill file)."""
    if KILL_FILE.exists():
        KILL_FILE.unlink()


def is_kill_switch_active() -> bool:
    """Check if kill switch is engaged."""
    return KILL_FILE.exists()


def get_control_pid() -> Optional[int]:
    """Get PID of running control service if any."""
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text().strip())
    except Exception:
        return None


def force_stop_control() -> bool:
    """Force stop any running control service."""
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
