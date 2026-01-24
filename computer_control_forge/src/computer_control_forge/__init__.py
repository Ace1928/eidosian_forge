"""
🎮 Computer Control Forge

Safe, auditable keyboard/mouse control and screen capture for Eidosian agents.

Safety mechanisms:
- Kill file check before every action
- PID file for process tracking
- Full provenance logging
- Rate limiting
"""

__version__ = "0.1.0"
__author__ = "Eidos"

from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent
KILL_FILE = Path("/tmp/eidosian_control_kill")
PID_FILE = Path("/tmp/eidosian_control.pid")


def is_kill_switch_active() -> bool:
    """Check if kill switch is engaged."""
    return KILL_FILE.exists()


def engage_kill_switch() -> None:
    """Engage the kill switch."""
    KILL_FILE.write_text("KILL")


def disengage_kill_switch() -> None:
    """Disengage the kill switch (use with caution)."""
    if KILL_FILE.exists():
        KILL_FILE.unlink()


# Lazy imports to avoid loading heavy dependencies until needed
def get_control_service():
    """Get the ControlService class (lazy import)."""
    from .control import ControlService
    return ControlService
