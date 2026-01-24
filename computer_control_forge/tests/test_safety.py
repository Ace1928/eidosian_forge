"""
Tests for safety mechanisms.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# Patch paths before import
TEST_KILL_FILE = Path(tempfile.gettempdir()) / "test_eidosian_control_kill"
TEST_PID_FILE = Path(tempfile.gettempdir()) / "test_eidosian_control.pid"


@pytest.fixture(autouse=True)
def clean_test_files():
    """Clean up test files before and after each test."""
    for f in [TEST_KILL_FILE, TEST_PID_FILE]:
        if f.exists():
            f.unlink()
    yield
    for f in [TEST_KILL_FILE, TEST_PID_FILE]:
        if f.exists():
            f.unlink()


def test_kill_switch_functions():
    """Test kill switch engage/disengage/check."""
    with patch("computer_control_forge.safety.KILL_FILE", TEST_KILL_FILE):
        from computer_control_forge.safety import (
            engage_kill_switch,
            disengage_kill_switch,
            is_kill_switch_active,
        )
        
        # Initially not active
        assert not is_kill_switch_active()
        
        # Engage
        engage_kill_switch()
        assert is_kill_switch_active()
        assert TEST_KILL_FILE.exists()
        
        # Disengage
        disengage_kill_switch()
        assert not is_kill_switch_active()
        assert not TEST_KILL_FILE.exists()


def test_safety_monitor_blocks_when_kill_active():
    """Test that SafetyMonitor raises when kill switch is active."""
    with patch("computer_control_forge.safety.KILL_FILE", TEST_KILL_FILE):
        with patch("computer_control_forge.safety.PID_FILE", TEST_PID_FILE):
            from computer_control_forge.safety import SafetyMonitor, KillSwitchEngaged
            
            # Create kill file first
            TEST_KILL_FILE.write_text("KILL")
            
            monitor = SafetyMonitor()
            
            with pytest.raises(KillSwitchEngaged):
                monitor.start()


def test_safety_monitor_writes_pid():
    """Test that SafetyMonitor writes PID file on start."""
    with patch("computer_control_forge.safety.KILL_FILE", TEST_KILL_FILE):
        with patch("computer_control_forge.safety.PID_FILE", TEST_PID_FILE):
            from computer_control_forge.safety import SafetyMonitor
            
            monitor = SafetyMonitor()
            monitor.start()
            
            assert TEST_PID_FILE.exists()
            assert int(TEST_PID_FILE.read_text()) == os.getpid()
            
            monitor.stop()
            assert not TEST_PID_FILE.exists()


def test_safety_monitor_check_raises_on_kill():
    """Test that check() raises after kill switch engaged."""
    with patch("computer_control_forge.safety.KILL_FILE", TEST_KILL_FILE):
        with patch("computer_control_forge.safety.PID_FILE", TEST_PID_FILE):
            from computer_control_forge.safety import SafetyMonitor, KillSwitchEngaged
            
            monitor = SafetyMonitor()
            monitor.start()
            
            # Engage kill switch while running
            TEST_KILL_FILE.write_text("KILL")
            
            with pytest.raises(KillSwitchEngaged):
                monitor.check()
            
            monitor.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
