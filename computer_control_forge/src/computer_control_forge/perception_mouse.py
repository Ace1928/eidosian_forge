#!/usr/bin/env python3
"""
Perception-Integrated Mouse Controller

Combines advanced mouse control with visual perception for:
- Edge-aware movement (stops at screen edges like humans)
- Visual verification of movements
- Target finding via OCR/window detection
- Intelligent path planning around obstacles

Author: Eidos
Version: 1.0.0
"""

import subprocess
import os
import time
import math
import random
from typing import Tuple, Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from datetime import datetime

# Local imports
try:
    from .cursor_position import KWinCursorPosition, CursorPositionError
    from .perception import (
        UnifiedPerception, PerceptionState, WindowInfo,
        BoundingBox, TextRegion
    )
except ImportError:
    from cursor_position import KWinCursorPosition, CursorPositionError
    from perception import (
        UnifiedPerception, PerceptionState, WindowInfo,
        BoundingBox, TextRegion
    )


@dataclass
class MovementResult:
    """Result of a mouse movement."""
    success: bool
    target: Tuple[int, int]
    actual: Tuple[int, int]
    
    # Error metrics
    error_distance: float
    error_x: int
    error_y: int
    
    # Timing
    duration_ms: float
    
    # Path info
    path_points: int
    hit_edge: bool = False
    edge_direction: Optional[str] = None
    
    # Perception context
    window_at_target: Optional[str] = None
    text_at_target: Optional[str] = None


@dataclass
class HumanProfile:
    """Human movement characteristics."""
    # Speed (pixels per second)
    min_speed: float = 200.0
    max_speed: float = 1500.0
    
    # Acceleration
    accel_time: float = 0.15      # Time to reach max speed
    decel_distance: float = 100.0  # Start slowing this far from target
    
    # Curve
    curve_amount: float = 0.15    # Path curvature (0-1)
    
    # Noise
    noise_amplitude: float = 1.5
    
    # Timing
    reaction_time_ms: float = 80.0
    
    # Overshoot
    overshoot_chance: float = 0.1
    overshoot_max: float = 0.05   # Max overshoot as fraction of distance


class PerceptionMouse:
    """
    Mouse controller with perception integration.
    
    Features:
    - Edge awareness: Detects and stops at screen edges
    - Visual feedback: Verifies movements via perception
    - Target finding: Find and click text/windows
    - Human-like motion: Curves, acceleration, noise
    """
    
    # Edge margins (pixels from screen edge)
    EDGE_MARGIN = 5
    
    # Movement tolerances
    POSITION_TOLERANCE = 8
    MAX_ITERATIONS = 25
    
    # Update rate
    UPDATE_HZ = 50
    
    def __init__(self, profile: Optional[HumanProfile] = None):
        self.profile = profile or HumanProfile()
        
        # Initialize perception
        self.perception = UnifiedPerception()
        
        # Cursor tracking
        self.cursor = KWinCursorPosition()
        
        # Screen bounds (from perception)
        self._update_screen_bounds()
        
        # Movement state
        self._movement_start: float = 0
        self._path: List[Tuple[int, int]] = []
        
        # Statistics
        self.stats = {
            "movements": 0,
            "successful": 0,
            "edge_hits": 0,
            "corrections": 0
        }
        
        os.environ.setdefault("YDOTOOL_SOCKET", "/tmp/.ydotool_socket")
    
    def _update_screen_bounds(self):
        """Update screen dimensions from perception."""
        state = self.perception.update()
        self.screen_width = state.screen_width
        self.screen_height = state.screen_height
    
    def get_position(self) -> Tuple[int, int]:
        """Get current cursor position."""
        return self.cursor.get_position()
    
    def _is_at_edge(self, x: int, y: int) -> Tuple[bool, Optional[str]]:
        """Check if position is at screen edge."""
        at_edge = False
        direction = None
        
        if x <= self.EDGE_MARGIN:
            at_edge = True
            direction = "left"
        elif x >= self.screen_width - self.EDGE_MARGIN:
            at_edge = True
            direction = "right"
        
        if y <= self.EDGE_MARGIN:
            at_edge = True
            direction = "top" if direction is None else f"{direction}_top"
        elif y >= self.screen_height - self.EDGE_MARGIN:
            at_edge = True
            direction = "bottom" if direction is None else f"{direction}_bottom"
        
        return at_edge, direction
    
    def _clamp_to_screen(self, x: int, y: int) -> Tuple[int, int]:
        """Clamp coordinates to valid screen area."""
        x = max(self.EDGE_MARGIN, min(x, self.screen_width - self.EDGE_MARGIN - 1))
        y = max(self.EDGE_MARGIN, min(y, self.screen_height - self.EDGE_MARGIN - 1))
        return x, y
    
    def _execute_move(self, dx: int, dy: int) -> bool:
        """Execute relative mouse movement."""
        if dx == 0 and dy == 0:
            return True
        
        result = subprocess.run(
            ["ydotool", "mousemove", "-x", str(dx), "-y", str(dy)],
            capture_output=True
        )
        return result.returncode == 0
    
    def _calculate_speed(self, distance: float, 
                         current_distance: float) -> float:
        """Calculate movement speed based on distance from target."""
        # Ease-out: slow down as we approach target
        if current_distance < self.profile.decel_distance:
            factor = current_distance / self.profile.decel_distance
            return self.profile.min_speed + (
                self.profile.max_speed - self.profile.min_speed
            ) * factor
        
        return self.profile.max_speed
    
    def _calculate_curve_offset(self, progress: float, 
                                 perpendicular: Tuple[float, float],
                                 total_distance: float) -> Tuple[float, float]:
        """Calculate curve offset at given progress."""
        # Sin wave for smooth curve
        curve_factor = math.sin(progress * math.pi) * self.profile.curve_amount
        offset = total_distance * curve_factor
        
        return (perpendicular[0] * offset, perpendicular[1] * offset)
    
    def move_to(self, target_x: int, target_y: int,
                human_like: bool = True,
                verify: bool = True) -> MovementResult:
        """
        Move cursor to target with perception integration.
        
        Args:
            target_x: Target X coordinate
            target_y: Target Y coordinate
            human_like: Use human-like motion characteristics
            verify: Verify final position and context
        """
        self.stats["movements"] += 1
        start_time = time.time()
        self._path = []
        hit_edge = False
        edge_direction = None
        
        # Clamp target to valid screen area
        target_x, target_y = self._clamp_to_screen(target_x, target_y)
        
        # Get starting position
        try:
            current_x, current_y = self.get_position()
        except CursorPositionError:
            return self._error_result((target_x, target_y), "Cannot get cursor position")
        
        self._path.append((current_x, current_y))
        start_pos = (current_x, current_y)
        
        # Calculate initial trajectory
        dx = target_x - current_x
        dy = target_y - current_y
        total_distance = math.sqrt(dx*dx + dy*dy)
        
        if total_distance < 2:
            # Already at target
            return self._success_result(start_pos, (target_x, target_y), 
                                        (current_x, current_y), start_time)
        
        # Perpendicular vector for curve
        if total_distance > 0:
            perp = (-dy / total_distance, dx / total_distance)
        else:
            perp = (0, 0)
        
        # Random curve direction
        curve_dir = random.choice([-1, 1])
        perp = (perp[0] * curve_dir, perp[1] * curve_dir)
        
        # Movement loop
        update_interval = 1.0 / self.UPDATE_HZ
        prev_x, prev_y = current_x, current_y
        
        for iteration in range(self.MAX_ITERATIONS):
            # Get current position
            try:
                current_x, current_y = self.get_position()
            except CursorPositionError:
                time.sleep(0.05)
                continue
            
            self._path.append((current_x, current_y))
            
            # Calculate error from target
            error_x = target_x - current_x
            error_y = target_y - current_y
            error_dist = math.sqrt(error_x*error_x + error_y*error_y)
            
            # Check if converged
            if error_dist <= self.POSITION_TOLERANCE:
                break
            
            # Smart edge detection: only if we're AT edge AND the target is beyond
            at_edge, edge_dir = self._is_at_edge(current_x, current_y)
            target_would_go_past_edge = False
            
            if at_edge and edge_dir:
                # Only count as "hitting edge" if target is beyond current edge position
                if "left" in edge_dir and target_x < current_x:
                    target_would_go_past_edge = True
                    target_x = max(target_x, current_x)  # Clamp target
                if "right" in edge_dir and target_x > current_x:
                    target_would_go_past_edge = True
                    target_x = min(target_x, current_x)
                if "top" in edge_dir and target_y < current_y:
                    target_would_go_past_edge = True
                    target_y = max(target_y, current_y)
                if "bottom" in edge_dir and target_y > current_y:
                    target_would_go_past_edge = True
                    target_y = min(target_y, current_y)
                
                if target_would_go_past_edge and not hit_edge:
                    # First time hitting this edge
                    hit_edge = True
                    edge_direction = edge_dir
                    self.stats["edge_hits"] += 1
                    
                    # Recalculate error with adjusted target
                    error_x = target_x - current_x
                    error_y = target_y - current_y
                    error_dist = math.sqrt(error_x*error_x + error_y*error_y)
                    
                    # May already be converged after target adjustment
                    if error_dist <= self.POSITION_TOLERANCE:
                        break
            
            # Calculate progress along path
            progress = 1.0 - (error_dist / max(1, total_distance))
            progress = max(0, min(1, progress))
            
            # Proportional control - works reliably with feedback
            # Gain varies: faster when far, slower when close (human-like)
            if error_dist > 100:
                gain = 0.5  # Aggressive when far
            elif error_dist > 30:
                gain = 0.4  # Medium distance
            else:
                gain = 0.35  # Precise when close
            
            # Base proportional movement
            base_dx = error_x * gain
            base_dy = error_y * gain
            
            # Add human-like curve during the main part of movement
            if human_like and progress < 0.7 and error_dist > 30:
                curve_offset = self._calculate_curve_offset(progress, perp, total_distance)
                curve_strength = (0.7 - progress) * 0.15  # Fade out curve
                base_dx += curve_offset[0] * curve_strength
                base_dy += curve_offset[1] * curve_strength
            
            # Add subtle noise when not close to target
            if human_like and error_dist > 25:
                noise_scale = min(2.0, error_dist / 50) * self.profile.noise_amplitude * 0.5
                base_dx += random.gauss(0, noise_scale)
                base_dy += random.gauss(0, noise_scale)
            
            # Round to integers
            final_dx = int(round(base_dx))
            final_dy = int(round(base_dy))
            
            # Ensure minimum movement
            if final_dx == 0 and abs(error_x) > 0:
                final_dx = 1 if error_x > 0 else -1
            if final_dy == 0 and abs(error_y) > 0:
                final_dy = 1 if error_y > 0 else -1
            
            # Execute
            self._execute_move(final_dx, final_dy)
            
            time.sleep(update_interval)
        
        # Final position verification
        try:
            final_x, final_y = self.get_position()
        except:
            final_x, final_y = current_x, current_y
        
        self._path.append((final_x, final_y))
        
        # Calculate final error
        final_error_x = abs(final_x - target_x)
        final_error_y = abs(final_y - target_y)
        final_error_dist = math.sqrt(final_error_x**2 + final_error_y**2)
        
        success = final_error_dist <= self.POSITION_TOLERANCE
        if success:
            self.stats["successful"] += 1
        
        # Get perception context if requested
        window_at_target = None
        text_at_target = None
        
        if verify:
            state = self.perception.update()
            
            # Window at cursor
            if state.window_under_cursor:
                window_at_target = state.window_under_cursor.caption
            
            # Nearby text
            cursor_context = self.perception.get_cursor_context()
            if cursor_context.get("nearby_text"):
                text_at_target = cursor_context["nearby_text"][0].text
        
        duration_ms = (time.time() - start_time) * 1000
        
        return MovementResult(
            success=success,
            target=(target_x, target_y),
            actual=(final_x, final_y),
            error_distance=final_error_dist,
            error_x=final_error_x,
            error_y=final_error_y,
            duration_ms=duration_ms,
            path_points=len(self._path),
            hit_edge=hit_edge,
            edge_direction=edge_direction,
            window_at_target=window_at_target,
            text_at_target=text_at_target
        )
    
    def _success_result(self, start: Tuple[int, int], 
                        target: Tuple[int, int],
                        actual: Tuple[int, int],
                        start_time: float) -> MovementResult:
        """Create success result."""
        self.stats["successful"] += 1
        return MovementResult(
            success=True,
            target=target,
            actual=actual,
            error_distance=0,
            error_x=0,
            error_y=0,
            duration_ms=(time.time() - start_time) * 1000,
            path_points=1
        )
    
    def _error_result(self, target: Tuple[int, int], 
                      error: str) -> MovementResult:
        """Create error result."""
        return MovementResult(
            success=False,
            target=target,
            actual=(0, 0),
            error_distance=float('inf'),
            error_x=0,
            error_y=0,
            duration_ms=0,
            path_points=0
        )
    
    # === High-level Actions ===
    
    def click(self, button: str = "left") -> bool:
        """Click at current position."""
        codes = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}
        result = subprocess.run(
            ["ydotool", "click", codes.get(button, "0xC0")],
            capture_output=True
        )
        return result.returncode == 0
    
    def click_at(self, x: int, y: int, button: str = "left") -> MovementResult:
        """Move to position and click."""
        result = self.move_to(x, y)
        if result.success:
            time.sleep(random.uniform(0.03, 0.08))
            self.click(button)
        return result
    
    def click_text(self, text: str, button: str = "left") -> Optional[MovementResult]:
        """Find text on screen and click it."""
        location = self.perception.find_text_location(text, refresh_ocr=True)
        if location:
            return self.click_at(*location, button)
        return None
    
    def click_window(self, title: str, button: str = "left") -> Optional[MovementResult]:
        """Find window by title and click its center."""
        window = self.perception.find_window(title)
        if window:
            return self.click_at(*window.center, button)
        return None
    
    def move_to_text(self, text: str) -> Optional[MovementResult]:
        """Move to text location."""
        location = self.perception.find_text_location(text, refresh_ocr=True)
        if location:
            return self.move_to(*location)
        return None
    
    def move_to_window(self, title: str) -> Optional[MovementResult]:
        """Move to window center."""
        window = self.perception.find_window(title)
        if window:
            return self.move_to(*window.center)
        return None
    
    def drag(self, start: Tuple[int, int], end: Tuple[int, int],
             button: str = "left") -> MovementResult:
        """Drag from start to end."""
        # Move to start
        result = self.move_to(*start)
        if not result.success:
            return result
        
        # Press button
        press_codes = {"left": "0x40", "right": "0x41", "middle": "0x42"}
        subprocess.run(["ydotool", "click", press_codes.get(button, "0x40")], 
                       capture_output=True)
        time.sleep(0.05)
        
        # Move to end
        result = self.move_to(*end)
        
        # Release button
        release_codes = {"left": "0x80", "right": "0x81", "middle": "0x82"}
        subprocess.run(["ydotool", "click", release_codes.get(button, "0x80")],
                       capture_output=True)
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics."""
        return {
            **self.stats,
            "success_rate": (
                self.stats["successful"] / max(1, self.stats["movements"])
            ),
            "perception_stats": self.perception.get_stats()
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.perception.cleanup()


# Convenience functions
def create_mouse() -> PerceptionMouse:
    """Create perception-integrated mouse controller."""
    return PerceptionMouse()


if __name__ == "__main__":
    print("=" * 70)
    print("Perception-Integrated Mouse Controller Test")
    print("=" * 70)
    
    mouse = PerceptionMouse()
    print(f"Screen: {mouse.screen_width}x{mouse.screen_height}")
    
    try:
        x, y = mouse.get_position()
        print(f"Current position: ({x}, {y})")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    
    targets = [
        (960, 540, "center"),
        (50, 50, "top-left (edge test)"),
        (1870, 50, "top-right (edge test)"),
        (1870, 1030, "bottom-right (edge test)"),
        (50, 1030, "bottom-left (edge test)"),
        (960, 540, "back to center"),
    ]
    
    print("\n" + "-" * 70)
    
    for tx, ty, name in targets:
        result = mouse.move_to(tx, ty)
        status = "✓" if result.success else "✗"
        edge_info = f" [EDGE:{result.edge_direction}]" if result.hit_edge else ""
        
        print(f"{status} {name}:")
        print(f"   Target: ({tx}, {ty}) -> Actual: {result.actual}")
        print(f"   Error: {result.error_distance:.1f}px, Time: {result.duration_ms:.0f}ms, "
              f"Path: {result.path_points} pts{edge_info}")
        
        if result.window_at_target:
            print(f"   Window: {result.window_at_target}")
        
        time.sleep(0.3)
    
    print("-" * 70)
    stats = mouse.get_stats()
    print(f"\nStats: {stats['successful']}/{stats['movements']} successful, "
          f"{stats['edge_hits']} edge hits")
    
    mouse.cleanup()
    print("=" * 70)
