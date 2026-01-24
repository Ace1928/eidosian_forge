"""
🔄 Autonomous Continuation Daemon

Provides capability for self-triggering continuation of Eidos sessions.
Monitors for specific states and can trigger input to continue iteration.

Safety Features:
- Global switch via control file
- Configurable idle timeout
- Activity logging
- Manual override capability

Created: 2026-01-23
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

# Control files
STOP_FILE = Path("/tmp/eidosian_continuation_stop")
STATE_FILE = Path("/tmp/eidosian_continuation_state.json")
LOG_FILE = Path("/tmp/eidosian_continuation.log")


def log_message(message: str) -> None:
    """Append message to log file."""
    timestamp = datetime.now(timezone.utc).isoformat()
    with LOG_FILE.open("a") as f:
        f.write(f"[{timestamp}] {message}\n")


@dataclass
class DaemonState:
    """State of the continuation daemon."""
    running: bool = False
    started_at: Optional[str] = None
    last_activity: Optional[str] = None
    triggers_sent: int = 0
    idle_threshold_sec: float = 300  # 5 minutes default
    
    def to_dict(self) -> Dict:
        return {
            "running": self.running,
            "started_at": self.started_at,
            "last_activity": self.last_activity,
            "triggers_sent": self.triggers_sent,
            "idle_threshold_sec": self.idle_threshold_sec
        }
    
    def save(self) -> None:
        STATE_FILE.write_text(json.dumps(self.to_dict(), indent=2))
    
    @classmethod
    def load(cls) -> "DaemonState":
        if STATE_FILE.exists():
            data = json.loads(STATE_FILE.read_text())
            return cls(**data)
        return cls()


def check_stop_switch() -> bool:
    """Check if stop switch is active."""
    return STOP_FILE.exists()


def activate_stop_switch() -> None:
    """Activate the stop switch to halt the daemon."""
    STOP_FILE.touch()
    log_message("Stop switch activated")


def deactivate_stop_switch() -> None:
    """Deactivate the stop switch."""
    if STOP_FILE.exists():
        STOP_FILE.unlink()
        log_message("Stop switch deactivated")


def get_status() -> Dict:
    """Get current daemon status."""
    state = DaemonState.load()
    return {
        **state.to_dict(),
        "stop_switch_active": check_stop_switch(),
        "state_file": str(STATE_FILE),
        "log_file": str(LOG_FILE),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# Simple trigger message
CONTINUATION_MESSAGE = """Good job Eidos! Review everything and proceed with next directions. Continue iterating indefinitely. Fully Eidosian."""


def type_continuation_message() -> bool:
    """Type the continuation message using ydotool."""
    from ..wayland_control import type_text
    
    if check_stop_switch():
        log_message("Continuation blocked: stop switch active")
        return False
    
    result = type_text(CONTINUATION_MESSAGE)
    success = result.get("success", False)
    
    if success:
        log_message(f"Typed continuation message ({len(CONTINUATION_MESSAGE)} chars)")
    else:
        log_message(f"Failed to type message: {result.get('error')}")
    
    return success


def run_daemon(idle_threshold_sec: float = 300, check_interval_sec: float = 30) -> None:
    """
    Run the continuation daemon.
    
    Monitors for idle state and triggers continuation when threshold is reached.
    
    Args:
        idle_threshold_sec: Seconds of idle before triggering (default 5 min)
        check_interval_sec: How often to check (default 30 sec)
    """
    state = DaemonState()
    state.running = True
    state.started_at = datetime.now(timezone.utc).isoformat()
    state.idle_threshold_sec = idle_threshold_sec
    state.save()
    
    log_message(f"Daemon started (idle_threshold={idle_threshold_sec}s)")
    
    # Track last screenshot hash
    from ..visual_feedback import ScreenState
    last_hash = None
    last_change_time = time.time()
    
    def signal_handler(signum, frame):
        log_message("Received signal, shutting down")
        state.running = False
        state.save()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        while state.running:
            # Check stop switch
            if check_stop_switch():
                log_message("Stop switch detected, stopping")
                break
            
            # Capture current state
            current = ScreenState.capture()
            if current:
                if last_hash != current.hash:
                    # Screen changed - activity detected
                    last_hash = current.hash
                    last_change_time = time.time()
                    state.last_activity = datetime.now(timezone.utc).isoformat()
                    state.save()
                else:
                    # No change - check if idle threshold reached
                    idle_time = time.time() - last_change_time
                    if idle_time >= idle_threshold_sec:
                        log_message(f"Idle threshold reached ({idle_time:.0f}s)")
                        
                        # Trigger continuation
                        if type_continuation_message():
                            state.triggers_sent += 1
                            last_change_time = time.time()  # Reset
                            state.save()
            
            time.sleep(check_interval_sec)
    
    except KeyboardInterrupt:
        log_message("Keyboard interrupt, stopping")
    finally:
        state.running = False
        state.save()
        log_message("Daemon stopped")


__all__ = [
    "check_stop_switch",
    "activate_stop_switch", 
    "deactivate_stop_switch",
    "get_status",
    "type_continuation_message",
    "run_daemon",
    "CONTINUATION_MESSAGE"
]
