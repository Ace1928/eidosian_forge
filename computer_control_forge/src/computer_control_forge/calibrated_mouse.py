#!/usr/bin/env python3
"""
Calibrated Mouse Control for KDE Wayland

This module provides accurate mouse positioning by:
1. Using KWin scripting to get actual cursor position
2. Computing the delta needed to reach target
3. Applying calibrated relative movements with acceleration compensation

Author: Eidos
Version: 1.0.0
"""

import subprocess
import os
import time
import math
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

# Handle both relative and absolute imports
try:
    from .cursor_position import get_cursor_position, KWinCursorPosition, CursorPositionError
except ImportError:
    from cursor_position import get_cursor_position, KWinCursorPosition, CursorPositionError


@dataclass
class MovementResult:
    """Result of a mouse movement operation."""
    success: bool
    target: Tuple[int, int]
    actual: Tuple[int, int]
    error: Tuple[int, int]
    attempts: int
    duration_ms: float
    
    @property
    def error_distance(self) -> float:
        """Euclidean distance from target."""
        return math.sqrt(self.error[0]**2 + self.error[1]**2)


class CalibratedMouse:
    """
    Calibrated mouse controller for KDE Wayland.
    
    Uses position feedback to achieve accurate positioning despite
    mouse acceleration being applied by libinput.
    """
    
    # Calibration constants
    ACCELERATION_FACTOR = 1.28
    MAX_ATTEMPTS = 5
    POSITION_TOLERANCE = 5
    MOVEMENT_DELAY = 0.1
    
    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.cursor_tracker = KWinCursorPosition()
        
        if "YDOTOOL_SOCKET" not in os.environ:
            os.environ["YDOTOOL_SOCKET"] = "/tmp/.ydotool_socket"
    
    def get_position(self) -> Tuple[int, int]:
        return self.cursor_tracker.get_position()
    
    def _ydotool_relative(self, dx: int, dy: int) -> bool:
        result = subprocess.run(
            ["ydotool", "mousemove", "-x", str(dx), "-y", str(dy)],
            capture_output=True
        )
        return result.returncode == 0
    
    def _compensate_acceleration(self, delta: int) -> int:
        if delta == 0:
            return 0
        compensated = int(delta / self.ACCELERATION_FACTOR)
        if compensated == 0:
            compensated = 1 if delta > 0 else -1
        return compensated
    
    def move_to(self, target_x: int, target_y: int, 
                tolerance: Optional[int] = None) -> MovementResult:
        start_time = time.time()
        tolerance = tolerance or self.POSITION_TOLERANCE
        
        target_x = max(0, min(target_x, self.screen_width - 1))
        target_y = max(0, min(target_y, self.screen_height - 1))
        
        attempts = 0
        
        for attempt in range(self.MAX_ATTEMPTS):
            attempts += 1
            
            try:
                current_x, current_y = self.get_position()
            except CursorPositionError:
                time.sleep(0.2)
                continue
            
            dx = target_x - current_x
            dy = target_y - current_y
            
            error_dist = math.sqrt(dx**2 + dy**2)
            if error_dist <= tolerance:
                return MovementResult(
                    success=True,
                    target=(target_x, target_y),
                    actual=(current_x, current_y),
                    error=(abs(dx), abs(dy)),
                    attempts=attempts,
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            # For fine-tuning, use raw values for small movements
            if attempt > 0 and abs(dx) < 20 and abs(dy) < 20:
                comp_dx, comp_dy = dx, dy
            else:
                comp_dx = self._compensate_acceleration(dx)
                comp_dy = self._compensate_acceleration(dy)
            
            self._ydotool_relative(comp_dx, comp_dy)
            time.sleep(self.MOVEMENT_DELAY)
        
        try:
            final_x, final_y = self.get_position()
        except CursorPositionError:
            final_x, final_y = 0, 0
        
        error_x = abs(final_x - target_x)
        error_y = abs(final_y - target_y)
        
        return MovementResult(
            success=error_x <= tolerance and error_y <= tolerance,
            target=(target_x, target_y),
            actual=(final_x, final_y),
            error=(error_x, error_y),
            attempts=attempts,
            duration_ms=(time.time() - start_time) * 1000
        )
    
    def click(self, button: str = "left") -> bool:
        button_codes = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}
        code = button_codes.get(button, "0xC0")
        result = subprocess.run(["ydotool", "click", code], capture_output=True)
        return result.returncode == 0
    
    def click_at(self, x: int, y: int, button: str = "left") -> MovementResult:
        result = self.move_to(x, y)
        if result.success:
            self.click(button)
        return result
