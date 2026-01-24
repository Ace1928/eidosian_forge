#!/usr/bin/env python3
"""
Feedback Loop Mouse Controller for KDE Wayland

Simple, stable feedback loop - measure, move fraction of error, repeat.
No complex PID - just proportional with damping.

Author: Eidos  
Version: 2.0.0
"""

import subprocess
import os
import time
import math
import random
from typing import Tuple, List
from dataclasses import dataclass, field

try:
    from .cursor_position import KWinCursorPosition, CursorPositionError
except ImportError:
    from cursor_position import KWinCursorPosition, CursorPositionError


@dataclass
class MovementResult:
    success: bool
    target: Tuple[int, int]
    actual: Tuple[int, int]
    error: float
    iterations: int
    duration_ms: float


class FeedbackMouse:
    """
    Simple feedback loop: measure error, move fraction toward target, repeat.
    
    The key insight: don't try to move the full distance at once.
    Move a fraction, measure, adjust. Let the loop converge naturally.
    """
    
    # How much of the error to correct each iteration (0-1)
    # Lower = smoother but slower, Higher = faster but may overshoot
    GAIN = 0.4
    
    # Slow down as we get close (prevents overshoot)
    FINE_GAIN = 0.6      # Use this gain when close to target
    FINE_THRESHOLD = 50  # "Close" = within this many pixels
    
    # Convergence  
    TOLERANCE = 5
    MAX_ITERATIONS = 30
    LOOP_DELAY = 0.025   # 40Hz update rate
    
    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.cursor = KWinCursorPosition()
        os.environ.setdefault("YDOTOOL_SOCKET", "/tmp/.ydotool_socket")
    
    def get_position(self) -> Tuple[int, int]:
        return self.cursor.get_position()
    
    def _move(self, dx: int, dy: int):
        """Execute relative move."""
        if dx == 0 and dy == 0:
            return
        subprocess.run(
            ["ydotool", "mousemove", "-x", str(dx), "-y", str(dy)],
            capture_output=True
        )
    
    def move_to(self, target_x: int, target_y: int) -> MovementResult:
        """Move to target using feedback loop."""
        start_time = time.time()
        
        target_x = max(0, min(target_x, self.screen_width - 1))
        target_y = max(0, min(target_y, self.screen_height - 1))
        
        for iteration in range(self.MAX_ITERATIONS):
            # Measure
            try:
                current_x, current_y = self.get_position()
            except CursorPositionError:
                time.sleep(0.05)
                continue
            
            # Calculate error
            error_x = target_x - current_x
            error_y = target_y - current_y
            error_dist = math.sqrt(error_x**2 + error_y**2)
            
            # Done?
            if error_dist <= self.TOLERANCE:
                return MovementResult(
                    True, (target_x, target_y), (current_x, current_y),
                    error_dist, iteration + 1, (time.time() - start_time) * 1000
                )
            
            # Choose gain based on distance (slow down when close)
            gain = self.FINE_GAIN if error_dist < self.FINE_THRESHOLD else self.GAIN
            
            # Calculate move (fraction of error)
            move_x = int(round(error_x * gain))
            move_y = int(round(error_y * gain))
            
            # Ensure we move at least 1px if there's error
            if move_x == 0 and abs(error_x) > 0:
                move_x = 1 if error_x > 0 else -1
            if move_y == 0 and abs(error_y) > 0:
                move_y = 1 if error_y > 0 else -1
            
            # Move
            self._move(move_x, move_y)
            time.sleep(self.LOOP_DELAY)
        
        # Final check
        try:
            final_x, final_y = self.get_position()
        except:
            final_x, final_y = 0, 0
        
        final_error = math.sqrt((target_x - final_x)**2 + (target_y - final_y)**2)
        
        return MovementResult(
            final_error <= self.TOLERANCE,
            (target_x, target_y), (final_x, final_y),
            final_error, self.MAX_ITERATIONS, (time.time() - start_time) * 1000
        )
    
    def click(self, button: str = "left") -> bool:
        codes = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}
        result = subprocess.run(["ydotool", "click", codes.get(button, "0xC0")], capture_output=True)
        return result.returncode == 0
    
    def click_at(self, x: int, y: int, button: str = "left") -> MovementResult:
        result = self.move_to(x, y)
        if result.success:
            time.sleep(0.05)
            self.click(button)
        return result


if __name__ == "__main__":
    print("=" * 60)
    print("Feedback Loop Mouse Controller")
    print("=" * 60)
    
    mouse = FeedbackMouse()
    
    try:
        x, y = mouse.get_position()
        print(f"Start: ({x}, {y})")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    
    print(f"Gain: {mouse.GAIN}, Fine: {mouse.FINE_GAIN}, Tolerance: {mouse.TOLERANCE}px\n")
    
    targets = [
        (960, 540, "center"),
        (100, 100, "top-left"),
        (1800, 100, "top-right"),
        (1800, 950, "bottom-right"),
        (100, 950, "bottom-left"),
        (960, 540, "center"),
    ]
    
    wins = 0
    for tx, ty, name in targets:
        result = mouse.move_to(tx, ty)
        ok = "✓" if result.success else "✗"
        wins += result.success
        print(f"{ok} {name:12} ({tx:4},{ty:4}) -> {result.actual} "
              f"err={result.error:.1f} iter={result.iterations} t={result.duration_ms:.0f}ms")
        time.sleep(0.2)
    
    print(f"\n{wins}/{len(targets)} successful")
