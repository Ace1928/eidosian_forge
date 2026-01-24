#!/usr/bin/env python3
"""
Cursor Position Utility for KDE Wayland

Uses KWin scripting API to get cursor position via D-Bus.
This is the only reliable way to get cursor position on KDE Wayland.

Author: Eidos
Version: 1.0.0
"""

import subprocess
import re
import time
from typing import Tuple, Optional
from pathlib import Path


class CursorPositionError(Exception):
    """Error getting cursor position."""
    pass


class KWinCursorPosition:
    """
    Get cursor position on KDE Wayland using KWin scripting API.
    
    Usage:
        cp = KWinCursorPosition()
        x, y = cp.get_position()
        print(f"Cursor at: ({x}, {y})")
    """
    
    SCRIPT_NAME = "eidos_cursor_probe"
    SCRIPT_CONTENT = 'console.log("EIDOS_CURSOR:" + workspace.cursorPos.x + "," + workspace.cursorPos.y);'
    CURSOR_PATTERN = re.compile(r'EIDOS_CURSOR:(\d+),(\d+)')
    
    def __init__(self, script_path: Optional[str] = None):
        """Initialize cursor position getter."""
        self._script_path = script_path or "/tmp/eidos_cursor_probe.js"
        self._ensure_script()
    
    def _ensure_script(self):
        """Ensure the probe script exists."""
        Path(self._script_path).write_text(self.SCRIPT_CONTENT)
    
    def _qdbus(self, *args) -> Tuple[bool, str]:
        """Run qdbus command and return (success, output)."""
        try:
            result = subprocess.run(
                ["qdbus", "org.kde.KWin", "/Scripting"] + list(args),
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0, result.stdout.strip()
        except Exception as e:
            return False, str(e)
    
    def get_position(self, timeout: float = 2.0) -> Tuple[int, int]:
        """
        Get current cursor position.
        
        Returns:
            Tuple of (x, y) coordinates
            
        Raises:
            CursorPositionError: If unable to get position
        """
        # Unload first in case it's already loaded
        self._qdbus("org.kde.kwin.Scripting.unloadScript", self.SCRIPT_NAME)
        
        # Load script
        success, result = self._qdbus(
            "org.kde.kwin.Scripting.loadScript",
            self._script_path,
            self.SCRIPT_NAME
        )
        if not success or not result.lstrip('-').isdigit() or result == "-1":
            raise CursorPositionError(f"Failed to load script: {result}")
        
        script_id = int(result)
        
        try:
            # Start script engine
            self._qdbus("org.kde.kwin.Scripting.start")
            
            # Wait briefly for script execution
            time.sleep(0.1)
            
            # Get position from journal
            position = self._parse_journal(timeout)
            return position
            
        finally:
            # Always unload script
            self._qdbus("org.kde.kwin.Scripting.unloadScript", self.SCRIPT_NAME)
    
    def _parse_journal(self, timeout: float) -> Tuple[int, int]:
        """Parse cursor position from journalctl output."""
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            try:
                # Get recent journal entries
                result = subprocess.run(
                    ["journalctl", "-n", "50", "--no-pager", "-o", "cat"],
                    capture_output=True, text=True, timeout=2
                )
                
                # Find our cursor position output
                for line in reversed(result.stdout.split('\n')):
                    match = self.CURSOR_PATTERN.search(line)
                    if match:
                        return int(match.group(1)), int(match.group(2))
                
            except Exception:
                pass
            
            time.sleep(0.1)
        
        raise CursorPositionError("Timeout waiting for cursor position from journal")


def get_cursor_position() -> Tuple[int, int]:
    """
    Convenience function to get cursor position.
    
    Returns:
        Tuple of (x, y) coordinates
    """
    cp = KWinCursorPosition()
    return cp.get_position()


if __name__ == "__main__":
    import sys
    
    print("=" * 50)
    print("KDE Wayland Cursor Position Utility")
    print("=" * 50)
    
    try:
        cp = KWinCursorPosition()
        x, y = cp.get_position()
        print(f"\n✓ Cursor position: ({x}, {y})")
        
        # If running interactively, show updates
        if len(sys.argv) > 1 and sys.argv[1] == "--watch":
            print("\nWatching cursor position (Ctrl+C to stop)...")
            try:
                while True:
                    x, y = cp.get_position()
                    print(f"\r  Position: ({x:4d}, {y:4d})", end="", flush=True)
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\n\nStopped.")
                
    except CursorPositionError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
