#!/usr/bin/env python3
"""
Human-like Mouse Control for KDE Wayland

Provides natural, human-mimicking mouse movements with:
- Bezier curve trajectories (not straight lines)
- Variable speed with ease-in-out
- Natural timing variations
- Acceleration compensation
- Screen-aware absolute positioning

Author: Eidos
Version: 1.1.0
"""

import subprocess
import os
import time
import math
import random
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

try:
    from .cursor_position import get_cursor_position, KWinCursorPosition, CursorPositionError
except ImportError:
    from cursor_position import get_cursor_position, KWinCursorPosition, CursorPositionError


@dataclass
class ScreenInfo:
    """Screen dimension information."""
    width: int
    height: int
    
    @classmethod
    def detect(cls) -> 'ScreenInfo':
        """Detect screen dimensions."""
        try:
            result = subprocess.run(
                ["xrandr", "--current"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                import re
                for line in result.stdout.split('\n'):
                    if ' connected' in line:
                        match = re.search(r'(\d+)x(\d+)', line)
                        if match:
                            return cls(int(match.group(1)), int(match.group(2)))
        except:
            pass
        return cls(1920, 1080)


@dataclass 
class MovementResult:
    """Result of a mouse movement."""
    success: bool
    start: Tuple[int, int]
    target: Tuple[int, int]
    actual: Tuple[int, int]
    error: Tuple[int, int]
    duration_ms: float
    path_points: int
    
    @property
    def error_distance(self) -> float:
        return math.sqrt(self.error[0]**2 + self.error[1]**2)


class HumanMouse:
    """
    Human-like mouse controller with acceleration compensation.
    """
    
    # Movement timing
    MIN_DURATION_MS = 200
    MAX_DURATION_MS = 1000
    PIXELS_PER_MS = 1.2
    
    # Human characteristics
    CURVE_VARIANCE = 0.25
    SPEED_VARIANCE = 0.15
    NOISE_AMPLITUDE = 1.5
    
    # Acceleration compensation (libinput typically accelerates by ~1.2-1.4x)
    ACCEL_FACTOR = 1.25
    
    # Accuracy
    TOLERANCE = 10
    MAX_CORRECTIONS = 3
    
    def __init__(self, screen: Optional[ScreenInfo] = None):
        self.screen = screen or ScreenInfo.detect()
        self.cursor = KWinCursorPosition()
        
        if "YDOTOOL_SOCKET" not in os.environ:
            os.environ["YDOTOOL_SOCKET"] = "/tmp/.ydotool_socket"
    
    def get_position(self) -> Tuple[int, int]:
        return self.cursor.get_position()
    
    def _compensate(self, delta: int) -> int:
        """Compensate for mouse acceleration."""
        if delta == 0:
            return 0
        # Under-request to compensate for acceleration
        compensated = int(delta / self.ACCEL_FACTOR)
        if compensated == 0:
            compensated = 1 if delta > 0 else -1
        return compensated
    
    def _bezier_point(self, t: float, p0: Tuple[float, float], 
                      p1: Tuple[float, float], p2: Tuple[float, float], 
                      p3: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate cubic Bezier point."""
        u = 1 - t
        return (
            u*u*u*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t*t*t*p3[0],
            u*u*u*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t*t*t*p3[1]
        )
    
    def _ease_in_out(self, t: float) -> float:
        """Smooth ease-in-out curve."""
        if t < 0.5:
            return 4 * t * t * t
        return 1 - pow(-2 * t + 2, 3) / 2
    
    def _generate_curve_targets(self, start: Tuple[int, int], 
                                 end: Tuple[int, int], 
                                 num_points: int) -> List[Tuple[int, int]]:
        """Generate curved path waypoints."""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 10:
            return [end]
        
        # Perpendicular for curve
        perp = (-dy, dx)
        perp_len = math.sqrt(perp[0]**2 + perp[1]**2) or 1
        perp = (perp[0]/perp_len, perp[1]/perp_len)
        
        # Random curve
        curve = distance * self.CURVE_VARIANCE * random.uniform(-0.5, 0.5)
        
        # Control points
        p0 = (float(start[0]), float(start[1]))
        p1 = (start[0] + dx*0.3 + perp[0]*curve, start[1] + dy*0.3 + perp[1]*curve)
        p2 = (start[0] + dx*0.7 + perp[0]*curve*0.5, start[1] + dy*0.7 + perp[1]*curve*0.5)
        p3 = (float(end[0]), float(end[1]))
        
        # Generate points
        points = []
        for i in range(1, num_points + 1):
            t = self._ease_in_out(i / num_points)
            bx, by = self._bezier_point(t, p0, p1, p2, p3)
            
            # Add slight noise (less at ends)
            if 0.1 < (i/num_points) < 0.9:
                bx += random.gauss(0, self.NOISE_AMPLITUDE)
                by += random.gauss(0, self.NOISE_AMPLITUDE)
            
            points.append((int(round(bx)), int(round(by))))
        
        return points
    
    def _execute_move(self, dx: int, dy: int) -> bool:
        """Execute single relative move with compensation."""
        comp_dx = self._compensate(dx)
        comp_dy = self._compensate(dy)
        
        if comp_dx == 0 and comp_dy == 0:
            return True
        
        result = subprocess.run(
            ["ydotool", "mousemove", "-x", str(comp_dx), "-y", str(comp_dy)],
            capture_output=True
        )
        return result.returncode == 0
    
    def move_to(self, target_x: int, target_y: int, 
                human_like: bool = True) -> MovementResult:
        """
        Move to absolute position with human-like motion.
        """
        start_time = time.time()
        
        # Clamp target
        target_x = max(0, min(target_x, self.screen.width - 1))
        target_y = max(0, min(target_y, self.screen.height - 1))
        
        try:
            start_x, start_y = self.get_position()
        except CursorPositionError:
            return MovementResult(False, (0,0), (target_x, target_y), 
                                  (0,0), (target_x, target_y), 0, 0)
        
        dx = target_x - start_x
        dy = target_y - start_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if human_like and distance > 20:
            # Calculate duration and points based on distance
            duration_ms = min(self.MAX_DURATION_MS, 
                            max(self.MIN_DURATION_MS, distance / self.PIXELS_PER_MS))
            
            # More points for longer distances (but not too many)
            num_points = max(3, min(15, int(distance / 80)))
            
            # Generate curved waypoints
            waypoints = self._generate_curve_targets(
                (start_x, start_y), (target_x, target_y), num_points
            )
            
            # Calculate delay between points
            point_delay = (duration_ms / 1000.0) / len(waypoints)
            
            # Execute path with position feedback
            current_x, current_y = start_x, start_y
            
            for i, (wx, wy) in enumerate(waypoints):
                # Get actual position periodically for correction
                if i % 3 == 0 and i > 0:
                    try:
                        current_x, current_y = self.get_position()
                    except:
                        pass
                
                # Calculate move needed
                move_dx = wx - current_x
                move_dy = wy - current_y
                
                self._execute_move(move_dx, move_dy)
                current_x, current_y = wx, wy  # Assume success for smooth motion
                
                # Human-like delay with variance
                delay = point_delay * (1 + random.uniform(-0.2, 0.2))
                time.sleep(delay)
            
            path_points = len(waypoints)
        else:
            # Direct movement for short distances
            self._execute_move(dx, dy)
            path_points = 1
        
        # Final position check and correction
        time.sleep(0.05)
        
        for correction in range(self.MAX_CORRECTIONS):
            try:
                actual_x, actual_y = self.get_position()
            except CursorPositionError:
                break
            
            error_x = target_x - actual_x
            error_y = target_y - actual_y
            
            if abs(error_x) <= self.TOLERANCE and abs(error_y) <= self.TOLERANCE:
                break
            
            # Correction move (slower, more precise)
            self._execute_move(error_x, error_y)
            time.sleep(0.08)
        
        # Final measurement
        try:
            actual_x, actual_y = self.get_position()
        except:
            actual_x, actual_y = target_x, target_y
        
        final_error_x = abs(actual_x - target_x)
        final_error_y = abs(actual_y - target_y)
        
        return MovementResult(
            success=final_error_x <= self.TOLERANCE and final_error_y <= self.TOLERANCE,
            start=(start_x, start_y),
            target=(target_x, target_y),
            actual=(actual_x, actual_y),
            error=(final_error_x, final_error_y),
            duration_ms=(time.time() - start_time) * 1000,
            path_points=path_points
        )
    
    def move_relative(self, dx: int, dy: int, human_like: bool = True) -> MovementResult:
        """Move by relative amount."""
        try:
            x, y = self.get_position()
            return self.move_to(x + dx, y + dy, human_like)
        except:
            return MovementResult(False, (0,0), (0,0), (0,0), (0,0), 0, 0)
    
    def click(self, button: str = "left") -> bool:
        """Click at current position."""
        codes = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}
        result = subprocess.run(
            ["ydotool", "click", codes.get(button, "0xC0")],
            capture_output=True
        )
        return result.returncode == 0
    
    def click_at(self, x: int, y: int, button: str = "left") -> MovementResult:
        """Move and click with human-like motion."""
        result = self.move_to(x, y)
        if result.success:
            time.sleep(random.uniform(0.05, 0.12))
            self.click(button)
        return result


if __name__ == "__main__":
    print("=" * 60)
    print("Human-like Mouse Control Test v1.1")
    print("=" * 60)
    
    mouse = HumanMouse()
    print(f"Screen: {mouse.screen.width}x{mouse.screen.height}")
    
    try:
        x, y = mouse.get_position()
        print(f"Current: ({x}, {y})")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    
    targets = [
        (960, 540, "center"),
        (200, 200, "top-left"),
        (1700, 200, "top-right"),
        (1700, 900, "bottom-right"),
        (200, 900, "bottom-left"),
        (960, 540, "center again"),
    ]
    
    print("\nHuman-like movements:")
    success_count = 0
    for tx, ty, name in targets:
        result = mouse.move_to(tx, ty)
        status = "✓" if result.success else "✗"
        if result.success:
            success_count += 1
        print(f"  {status} {name}: target=({tx},{ty}) actual={result.actual} "
              f"err={result.error_distance:.1f}px pts={result.path_points} "
              f"time={result.duration_ms:.0f}ms")
        time.sleep(0.2)
    
    print(f"\nSuccess: {success_count}/{len(targets)}")
    print("=" * 60)
