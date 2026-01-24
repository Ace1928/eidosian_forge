#!/usr/bin/env python3
"""
Advanced Human-like Mouse Controller with PID Feedback

A sophisticated mouse control system combining:
- Dual-phase movement (ballistic + corrective)
- Bezier curve trajectories with natural variance
- Adaptive PID feedback loop with anti-windup
- Fitts's Law timing model
- Human biomechanical simulation (tremor, fatigue, reaction time)
- Velocity profiling with minimum-jerk optimization
- Stochastic noise modeling
- Automatic acceleration learning/compensation

Author: Eidos
Version: 3.0.0 - Fully Eidosian
"""

import subprocess
import os
import time
import math
import random
from typing import Tuple, Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

try:
    from .cursor_position import KWinCursorPosition, CursorPositionError
except ImportError:
    from cursor_position import KWinCursorPosition, CursorPositionError


# ============================================================================
# Data Structures
# ============================================================================

class MovementPhase(Enum):
    """Phases of human mouse movement."""
    REACTION = "reaction"      # Initial delay before movement
    BALLISTIC = "ballistic"    # Fast, open-loop movement toward target
    CORRECTIVE = "corrective"  # Slower, closed-loop fine adjustment
    VERIFY = "verify"          # Final position verification


@dataclass
class ScreenGeometry:
    """Screen information."""
    width: int
    height: int
    dpi: float = 96.0
    
    @classmethod
    def detect(cls) -> 'ScreenGeometry':
        try:
            import re
            result = subprocess.run(["xrandr"], capture_output=True, text=True, timeout=5)
            for line in result.stdout.split('\n'):
                if ' connected' in line:
                    match = re.search(r'(\d+)x(\d+)', line)
                    if match:
                        return cls(int(match.group(1)), int(match.group(2)))
        except:
            pass
        return cls(1920, 1080)


@dataclass
class PIDState:
    """State for PID controller with anti-windup."""
    integral: Tuple[float, float] = (0.0, 0.0)
    last_error: Tuple[float, float] = (0.0, 0.0)
    last_derivative: Tuple[float, float] = (0.0, 0.0)
    last_output: Tuple[float, float] = (0.0, 0.0)
    last_time: float = 0.0
    saturated: bool = False


@dataclass
class HumanProfile:
    """
    Human biomechanical parameters.
    Based on research on human motor control.
    """
    # Reaction time (ms) - time before movement starts
    reaction_time_mean: float = 180.0
    reaction_time_std: float = 40.0
    
    # Movement timing (Fitts's Law parameters)
    fitts_a: float = 50.0   # Base time (ms)
    fitts_b: float = 150.0  # Scaling factor
    
    # Tremor (physiological hand tremor ~8-12Hz)
    tremor_frequency: float = 10.0  # Hz
    tremor_amplitude: float = 0.8   # pixels
    
    # Noise characteristics
    spatial_noise_std: float = 2.0   # pixels
    temporal_noise_std: float = 0.15  # fraction of delay
    
    # Overshoot tendency
    overshoot_probability: float = 0.20
    overshoot_magnitude: float = 0.08  # fraction of distance
    
    # Fatigue (increases noise over time)
    fatigue_rate: float = 0.001  # per second
    
    # Sub-movement characteristics
    submovements_min: int = 2
    submovements_max: int = 5
    
    # Curve tendency (humans don't move in straight lines)
    curve_bias: float = 0.15  # perpendicular deviation as fraction of distance


@dataclass
class MovementMetrics:
    """Detailed metrics for a movement."""
    success: bool
    target: Tuple[int, int]
    actual: Tuple[int, int]
    start: Tuple[int, int]
    
    # Accuracy
    error_euclidean: float
    error_x: int
    error_y: int
    
    # Timing
    total_duration_ms: float
    reaction_time_ms: float
    ballistic_duration_ms: float
    corrective_duration_ms: float
    
    # Path characteristics
    path_length: float
    path_points: int
    path_efficiency: float  # straight_line / actual_path
    
    # Feedback loop stats
    pid_iterations: int
    corrections_made: int
    max_overshoot: float
    
    # Human-like characteristics applied
    tremor_applied: bool
    curve_applied: bool
    overshoot_occurred: bool
    
    # Raw path for analysis
    path: List[Tuple[int, int]] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)


# ============================================================================
# Mathematical Utilities
# ============================================================================

class MathUtils:
    """Mathematical utilities for movement calculations."""
    
    @staticmethod
    def bezier_cubic(t: float, p0: Tuple[float, float], p1: Tuple[float, float],
                     p2: Tuple[float, float], p3: Tuple[float, float]) -> Tuple[float, float]:
        """Evaluate cubic Bezier curve at parameter t."""
        u = 1.0 - t
        tt = t * t
        uu = u * u
        uuu = uu * u
        ttt = tt * t
        
        x = uuu * p0[0] + 3 * uu * t * p1[0] + 3 * u * tt * p2[0] + ttt * p3[0]
        y = uuu * p0[1] + 3 * uu * t * p1[1] + 3 * u * tt * p2[1] + ttt * p3[1]
        return (x, y)
    
    @staticmethod
    def bezier_derivative(t: float, p0: Tuple[float, float], p1: Tuple[float, float],
                          p2: Tuple[float, float], p3: Tuple[float, float]) -> Tuple[float, float]:
        """First derivative of cubic Bezier (velocity)."""
        u = 1.0 - t
        dx = 3 * u * u * (p1[0] - p0[0]) + 6 * u * t * (p2[0] - p1[0]) + 3 * t * t * (p3[0] - p2[0])
        dy = 3 * u * u * (p1[1] - p0[1]) + 6 * u * t * (p2[1] - p1[1]) + 3 * t * t * (p3[1] - p2[1])
        return (dx, dy)
    
    @staticmethod
    def minimum_jerk(t: float) -> float:
        """
        Minimum-jerk trajectory profile.
        This is the velocity profile that minimizes jerk (derivative of acceleration).
        Humans naturally produce movements close to minimum-jerk.
        """
        # Normalized time (0 to 1)
        return 10 * (t ** 3) - 15 * (t ** 4) + 6 * (t ** 5)
    
    @staticmethod
    def minimum_jerk_velocity(t: float) -> float:
        """Velocity profile for minimum-jerk trajectory."""
        return 30 * (t ** 2) - 60 * (t ** 3) + 30 * (t ** 4)
    
    @staticmethod
    def fitts_time(distance: float, width: float, a: float = 50, b: float = 150) -> float:
        """
        Fitts's Law: predict movement time.
        MT = a + b * log2(2D/W)
        """
        if width <= 0:
            width = 1
        index_of_difficulty = math.log2(2 * distance / width + 1)
        return a + b * index_of_difficulty
    
    @staticmethod
    def gaussian_noise(mean: float = 0.0, std: float = 1.0) -> float:
        """Generate Gaussian noise."""
        return random.gauss(mean, std)
    
    @staticmethod
    def perlin_noise_1d(x: float, octaves: int = 4) -> float:
        """Simple 1D Perlin-like noise for smooth randomness."""
        total = 0.0
        frequency = 1.0
        amplitude = 1.0
        max_value = 0.0
        
        for _ in range(octaves):
            # Simple interpolated noise
            xi = int(x * frequency)
            xf = (x * frequency) - xi
            
            # Hash-based pseudo-random
            n0 = math.sin(xi * 12.9898) * 43758.5453
            n1 = math.sin((xi + 1) * 12.9898) * 43758.5453
            n0 = n0 - int(n0)
            n1 = n1 - int(n1)
            
            # Smoothstep interpolation
            t = xf * xf * (3 - 2 * xf)
            total += (n0 * (1 - t) + n1 * t) * amplitude
            
            max_value += amplitude
            amplitude *= 0.5
            frequency *= 2.0
        
        return total / max_value
    
    @staticmethod
    def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Euclidean distance between two points."""
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    
    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp value to range."""
        return max(min_val, min(max_val, value))
    
    @staticmethod
    def lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation."""
        return a + (b - a) * t


# ============================================================================
# PID Controller
# ============================================================================

class AdaptivePIDController:
    """
    Advanced PID controller with:
    - Anti-windup (integral clamping and back-calculation)
    - Derivative filtering (low-pass to reduce noise)
    - Adaptive gain scheduling
    - Setpoint weighting
    """
    
    def __init__(self, kp: float = 0.5, ki: float = 0.02, kd: float = 0.1):
        # Base gains
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # Anti-windup
        self.integral_limit = 150.0
        self.output_limit = 80.0
        
        # Derivative filter coefficient (0-1, lower = more filtering)
        self.derivative_filter = 0.3
        
        # Setpoint weighting (reduces overshoot)
        self.setpoint_weight_p = 1.0
        self.setpoint_weight_d = 0.5
        
        # State
        self.state = PIDState()
    
    def reset(self):
        """Reset controller state."""
        self.state = PIDState()
        self.state.last_time = time.time()
    
    def compute(self, setpoint: Tuple[float, float], 
                measured: Tuple[float, float]) -> Tuple[float, float]:
        """
        Compute PID output for both axes.
        """
        current_time = time.time()
        dt = current_time - self.state.last_time if self.state.last_time > 0 else 0.02
        dt = max(0.001, min(dt, 0.1))  # Clamp dt to reasonable range
        
        # Calculate error
        error_x = setpoint[0] - measured[0]
        error_y = setpoint[1] - measured[1]
        
        # Distance-based gain scheduling (less aggressive when close)
        error_dist = math.sqrt(error_x ** 2 + error_y ** 2)
        gain_scale = MathUtils.clamp(error_dist / 100.0, 0.3, 1.0)
        
        # Proportional term (with setpoint weighting)
        p_x = self.kp * gain_scale * (self.setpoint_weight_p * setpoint[0] - measured[0])
        p_y = self.kp * gain_scale * (self.setpoint_weight_p * setpoint[1] - measured[1])
        
        # Integral term with anti-windup
        if not self.state.saturated:
            new_int_x = self.state.integral[0] + error_x * dt
            new_int_y = self.state.integral[1] + error_y * dt
            
            # Clamp integral
            new_int_x = MathUtils.clamp(new_int_x, -self.integral_limit, self.integral_limit)
            new_int_y = MathUtils.clamp(new_int_y, -self.integral_limit, self.integral_limit)
            
            self.state.integral = (new_int_x, new_int_y)
        
        i_x = self.ki * self.state.integral[0]
        i_y = self.ki * self.state.integral[1]
        
        # Derivative term with filtering and setpoint weighting
        if dt > 0:
            # Use filtered derivative
            raw_deriv_x = (error_x - self.state.last_error[0]) / dt
            raw_deriv_y = (error_y - self.state.last_error[1]) / dt
            
            # Low-pass filter
            filtered_deriv_x = (self.derivative_filter * raw_deriv_x + 
                               (1 - self.derivative_filter) * self.state.last_derivative[0])
            filtered_deriv_y = (self.derivative_filter * raw_deriv_y + 
                               (1 - self.derivative_filter) * self.state.last_derivative[1])
            
            self.state.last_derivative = (filtered_deriv_x, filtered_deriv_y)
            
            d_x = self.kd * self.setpoint_weight_d * filtered_deriv_x
            d_y = self.kd * self.setpoint_weight_d * filtered_deriv_y
        else:
            d_x = d_y = 0.0
        
        # Combine terms
        output_x = p_x + i_x + d_x
        output_y = p_y + i_y + d_y
        
        # Output limiting with saturation tracking
        output_magnitude = math.sqrt(output_x ** 2 + output_y ** 2)
        if output_magnitude > self.output_limit:
            scale = self.output_limit / output_magnitude
            output_x *= scale
            output_y *= scale
            self.state.saturated = True
        else:
            self.state.saturated = False
        
        # Update state
        self.state.last_error = (error_x, error_y)
        self.state.last_output = (output_x, output_y)
        self.state.last_time = current_time
        
        return (output_x, output_y)


# ============================================================================
# Trajectory Generator
# ============================================================================

class HumanTrajectoryGenerator:
    """
    Generates human-like mouse trajectories using:
    - Bezier curves with natural curvature
    - Minimum-jerk velocity profile
    - Sub-movement modeling
    - Stochastic perturbations
    """
    
    def __init__(self, profile: HumanProfile):
        self.profile = profile
        self.movement_start_time = 0.0
    
    def generate_control_points(self, start: Tuple[float, float], 
                                  end: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Generate Bezier control points for human-like curve."""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance < 1:
            return [start, start, end, end]
        
        # Perpendicular vector for curve
        perp_x = -dy / distance
        perp_y = dx / distance
        
        # Random curve bias (humans tend to curve movements)
        curve_amount = distance * self.profile.curve_bias
        curve_direction = random.choice([-1, 1]) * random.uniform(0.5, 1.5)
        
        # Control point 1: ~30% along path, curved
        t1 = random.uniform(0.25, 0.35)
        cp1 = (
            start[0] + dx * t1 + perp_x * curve_amount * curve_direction,
            start[1] + dy * t1 + perp_y * curve_amount * curve_direction
        )
        
        # Control point 2: ~70% along path, less curved (straightening toward target)
        t2 = random.uniform(0.65, 0.75)
        cp2 = (
            start[0] + dx * t2 + perp_x * curve_amount * curve_direction * 0.3,
            start[1] + dy * t2 + perp_y * curve_amount * curve_direction * 0.3
        )
        
        return [start, cp1, cp2, end]
    
    def generate_trajectory(self, start: Tuple[int, int], end: Tuple[int, int],
                            duration_ms: float) -> List[Tuple[float, float, float]]:
        """
        Generate trajectory as list of (x, y, timestamp_ms) points.
        Uses minimum-jerk profile with human-like perturbations.
        """
        start_f = (float(start[0]), float(start[1]))
        end_f = (float(end[0]), float(end[1]))
        distance = MathUtils.distance(start_f, end_f)
        
        if distance < 5:
            return [(end_f[0], end_f[1], duration_ms)]
        
        # Generate control points
        cps = self.generate_control_points(start_f, end_f)
        
        # Determine number of sub-movements
        num_submovements = random.randint(
            self.profile.submovements_min,
            self.profile.submovements_max
        )
        
        # Points per sub-movement (more points = smoother)
        points_per_sub = max(3, int(distance / 50))
        total_points = num_submovements * points_per_sub
        
        trajectory = []
        
        for i in range(total_points):
            # Normalized time
            t_normalized = (i + 1) / total_points
            
            # Apply minimum-jerk profile
            t_jerk = MathUtils.minimum_jerk(t_normalized)
            
            # Get point on Bezier curve
            point = MathUtils.bezier_cubic(t_jerk, cps[0], cps[1], cps[2], cps[3])
            
            # Add spatial noise (less at endpoints)
            endpoint_factor = 1.0 - abs(2 * t_normalized - 1) ** 2  # 0 at ends, 1 in middle
            noise_x = MathUtils.gaussian_noise(0, self.profile.spatial_noise_std * endpoint_factor)
            noise_y = MathUtils.gaussian_noise(0, self.profile.spatial_noise_std * endpoint_factor)
            
            # Add physiological tremor
            tremor_phase = 2 * math.pi * self.profile.tremor_frequency * (t_normalized * duration_ms / 1000)
            tremor_x = self.profile.tremor_amplitude * math.sin(tremor_phase + random.uniform(0, 0.5))
            tremor_y = self.profile.tremor_amplitude * math.cos(tremor_phase + random.uniform(0, 0.5))
            
            # Combine
            final_x = point[0] + noise_x + tremor_x
            final_y = point[1] + noise_y + tremor_y
            
            # Timestamp with temporal noise
            base_time = t_normalized * duration_ms
            time_noise = MathUtils.gaussian_noise(0, duration_ms * self.profile.temporal_noise_std / total_points)
            timestamp = max(0, base_time + time_noise)
            
            trajectory.append((final_x, final_y, timestamp))
        
        # Ensure final point is exactly the target
        trajectory[-1] = (end_f[0], end_f[1], duration_ms)
        
        return trajectory
    
    def should_overshoot(self, distance: float) -> Tuple[bool, float]:
        """Determine if this movement should overshoot and by how much."""
        if distance < 50:  # No overshoot for short movements
            return False, 0.0
        
        if random.random() < self.profile.overshoot_probability:
            magnitude = distance * self.profile.overshoot_magnitude * random.uniform(0.5, 1.5)
            return True, magnitude
        
        return False, 0.0


# ============================================================================
# Main Controller
# ============================================================================

class AdvancedMouseController:
    """
    Advanced human-like mouse controller.
    
    Combines trajectory generation with PID feedback for accurate,
    natural-looking mouse movements.
    """
    
    def __init__(self, screen: Optional[ScreenGeometry] = None,
                 profile: Optional[HumanProfile] = None):
        self.screen = screen or ScreenGeometry.detect()
        self.profile = profile or HumanProfile()
        
        self.cursor = KWinCursorPosition()
        self.pid = AdaptivePIDController(kp=0.45, ki=0.015, kd=0.08)
        self.trajectory_gen = HumanTrajectoryGenerator(self.profile)
        
        # Acceleration learning (adapts over time)
        self.acceleration_estimate = 1.0
        self.acceleration_samples: List[float] = []
        
        # State
        self.movement_count = 0
        self.fatigue_accumulated = 0.0
        self.session_start = time.time()
        
        # Settings
        self.tolerance = 5.0
        self.max_corrective_iterations = 15
        self.update_rate_hz = 60
        
        os.environ.setdefault("YDOTOOL_SOCKET", "/tmp/.ydotool_socket")
    
    def get_position(self) -> Tuple[int, int]:
        """Get current cursor position."""
        return self.cursor.get_position()
    
    def _execute_move(self, dx: float, dy: float) -> bool:
        """Execute relative mouse movement."""
        # Apply learned acceleration compensation
        comp_dx = int(round(dx / self.acceleration_estimate))
        comp_dy = int(round(dy / self.acceleration_estimate))
        
        if comp_dx == 0 and comp_dy == 0:
            if abs(dx) > 0.5:
                comp_dx = 1 if dx > 0 else -1
            if abs(dy) > 0.5:
                comp_dy = 1 if dy > 0 else -1
        
        if comp_dx == 0 and comp_dy == 0:
            return True
        
        result = subprocess.run(
            ["ydotool", "mousemove", "-x", str(comp_dx), "-y", str(comp_dy)],
            capture_output=True
        )
        return result.returncode == 0
    
    def _update_acceleration_estimate(self, intended: Tuple[float, float],
                                        actual: Tuple[float, float]):
        """Learn acceleration factor from movement results."""
        int_dist = math.sqrt(intended[0] ** 2 + intended[1] ** 2)
        act_dist = math.sqrt(actual[0] ** 2 + actual[1] ** 2)
        
        if int_dist > 10 and act_dist > 5:
            ratio = act_dist / int_dist
            self.acceleration_samples.append(ratio)
            
            # Keep last 20 samples
            if len(self.acceleration_samples) > 20:
                self.acceleration_samples.pop(0)
            
            # Update estimate (exponential moving average)
            if self.acceleration_samples:
                avg = sum(self.acceleration_samples) / len(self.acceleration_samples)
                self.acceleration_estimate = 0.8 * self.acceleration_estimate + 0.2 * avg
    
    def _apply_fatigue(self):
        """Apply fatigue effects to the human profile."""
        elapsed = time.time() - self.session_start
        self.fatigue_accumulated = min(1.0, elapsed * self.profile.fatigue_rate)
        
        # Increase noise slightly with fatigue
        # (Actual profile modification would be done here)
    
    def _reaction_delay(self) -> float:
        """Generate human-like reaction delay."""
        delay_ms = max(0, MathUtils.gaussian_noise(
            self.profile.reaction_time_mean,
            self.profile.reaction_time_std
        ))
        return delay_ms / 1000.0
    
    def move_to(self, target_x: int, target_y: int) -> MovementMetrics:
        """
        Move cursor to target with full human-like simulation.
        """
        movement_start = time.time()
        self.movement_count += 1
        self._apply_fatigue()
        
        # Clamp target to screen
        target_x = MathUtils.clamp(target_x, 0, self.screen.width - 1)
        target_y = MathUtils.clamp(target_y, 0, self.screen.height - 1)
        target = (float(target_x), float(target_y))
        
        # Get starting position
        try:
            start_x, start_y = self.get_position()
        except CursorPositionError as e:
            return self._error_result(target, str(e))
        
        start = (float(start_x), float(start_y))
        distance = MathUtils.distance(start, target)
        
        # Initialize metrics
        path = [(start_x, start_y)]
        timestamps = [0.0]
        corrections = 0
        max_overshoot = 0.0
        overshoot_occurred = False
        
        # ====== PHASE 1: REACTION TIME ======
        reaction_start = time.time()
        reaction_delay = self._reaction_delay()
        time.sleep(reaction_delay)
        reaction_time_ms = (time.time() - reaction_start) * 1000
        
        # ====== PHASE 2: BALLISTIC MOVEMENT ======
        ballistic_start = time.time()
        
        if distance > 10:
            # Calculate movement duration using Fitts's Law
            duration_ms = MathUtils.fitts_time(
                distance, self.tolerance * 2,
                self.profile.fitts_a, self.profile.fitts_b
            )
            
            # Add some randomness
            duration_ms *= random.uniform(0.9, 1.1)
            
            # Check for overshoot
            should_overshoot, overshoot_amount = self.trajectory_gen.should_overshoot(distance)
            
            if should_overshoot:
                # Calculate overshoot target
                dx = target[0] - start[0]
                dy = target[1] - start[1]
                norm = distance if distance > 0 else 1
                overshoot_target = (
                    target[0] + (dx / norm) * overshoot_amount,
                    target[1] + (dy / norm) * overshoot_amount
                )
                overshoot_occurred = True
            else:
                overshoot_target = target
            
            # Generate trajectory
            trajectory = self.trajectory_gen.generate_trajectory(
                (start_x, start_y),
                (int(overshoot_target[0]), int(overshoot_target[1])),
                duration_ms
            )
            
            # Execute ballistic trajectory
            current_x, current_y = float(start_x), float(start_y)
            traj_start_time = time.time()
            
            for tx, ty, t_ms in trajectory:
                # Calculate intended movement
                dx = tx - current_x
                dy = ty - current_y
                
                self._execute_move(dx, dy)
                
                # Brief delay based on trajectory timing
                elapsed_ms = (time.time() - traj_start_time) * 1000
                if t_ms > elapsed_ms:
                    time.sleep((t_ms - elapsed_ms) / 1000.0)
                
                # Update position estimate (we'll verify in corrective phase)
                current_x, current_y = tx, ty
                path.append((int(tx), int(ty)))
                timestamps.append((time.time() - movement_start) * 1000)
            
            # Track overshoot
            if overshoot_occurred:
                max_overshoot = overshoot_amount
        
        ballistic_duration_ms = (time.time() - ballistic_start) * 1000
        
        # ====== PHASE 3: CORRECTIVE MOVEMENT ======
        corrective_start = time.time()
        self.pid.reset()
        
        # Corrective sub-movements with feedback
        loop_delay = 1.0 / self.update_rate_hz
        
        for iteration in range(self.max_corrective_iterations):
            # Measure actual position
            try:
                actual_x, actual_y = self.get_position()
            except CursorPositionError:
                time.sleep(0.05)
                continue
            
            path.append((actual_x, actual_y))
            timestamps.append((time.time() - movement_start) * 1000)
            
            # Calculate error
            error_x = target_x - actual_x
            error_y = target_y - actual_y
            error_dist = math.sqrt(error_x ** 2 + error_y ** 2)
            
            # Track overshoot
            if error_dist > max_overshoot:
                max_overshoot = error_dist
            
            # Check convergence
            if error_dist <= self.tolerance:
                break
            
            # PID correction
            correction = self.pid.compute(target, (float(actual_x), float(actual_y)))
            
            # Execute correction with slight human noise
            noise_x = MathUtils.gaussian_noise(0, 0.5)
            noise_y = MathUtils.gaussian_noise(0, 0.5)
            
            self._execute_move(correction[0] + noise_x, correction[1] + noise_y)
            corrections += 1
            
            time.sleep(loop_delay)
        
        corrective_duration_ms = (time.time() - corrective_start) * 1000
        
        # ====== PHASE 4: VERIFY ======
        try:
            final_x, final_y = self.get_position()
        except CursorPositionError:
            final_x, final_y = path[-1] if path else (start_x, start_y)
        
        path.append((final_x, final_y))
        timestamps.append((time.time() - movement_start) * 1000)
        
        # Calculate final metrics
        error_x = abs(target_x - final_x)
        error_y = abs(target_y - final_y)
        error_euclidean = math.sqrt(error_x ** 2 + error_y ** 2)
        
        # Path length
        path_length = sum(
            MathUtils.distance(path[i], path[i + 1])
            for i in range(len(path) - 1)
        )
        
        # Path efficiency
        straight_line = MathUtils.distance(start, (final_x, final_y))
        path_efficiency = straight_line / path_length if path_length > 0 else 1.0
        
        total_duration_ms = (time.time() - movement_start) * 1000
        
        return MovementMetrics(
            success=error_euclidean <= self.tolerance,
            target=(target_x, target_y),
            actual=(final_x, final_y),
            start=(start_x, start_y),
            error_euclidean=error_euclidean,
            error_x=error_x,
            error_y=error_y,
            total_duration_ms=total_duration_ms,
            reaction_time_ms=reaction_time_ms,
            ballistic_duration_ms=ballistic_duration_ms,
            corrective_duration_ms=corrective_duration_ms,
            path_length=path_length,
            path_points=len(path),
            path_efficiency=path_efficiency,
            pid_iterations=corrections,
            corrections_made=corrections,
            max_overshoot=max_overshoot,
            tremor_applied=True,
            curve_applied=True,
            overshoot_occurred=overshoot_occurred,
            path=path,
            timestamps=timestamps
        )
    
    def _error_result(self, target: Tuple[float, float], error: str) -> MovementMetrics:
        """Create error result."""
        return MovementMetrics(
            success=False, target=(int(target[0]), int(target[1])),
            actual=(0, 0), start=(0, 0), error_euclidean=float('inf'),
            error_x=0, error_y=0, total_duration_ms=0, reaction_time_ms=0,
            ballistic_duration_ms=0, corrective_duration_ms=0, path_length=0,
            path_points=0, path_efficiency=0, pid_iterations=0, corrections_made=0,
            max_overshoot=0, tremor_applied=False, curve_applied=False,
            overshoot_occurred=False
        )
    
    def click(self, button: str = "left") -> bool:
        """Click at current position."""
        codes = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}
        result = subprocess.run(
            ["ydotool", "click", codes.get(button, "0xC0")],
            capture_output=True
        )
        return result.returncode == 0
    
    def click_at(self, x: int, y: int, button: str = "left") -> MovementMetrics:
        """Move to position and click."""
        result = self.move_to(x, y)
        if result.success:
            # Human-like delay before click
            time.sleep(random.uniform(0.03, 0.08))
            self.click(button)
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics."""
        return {
            "movement_count": self.movement_count,
            "acceleration_estimate": self.acceleration_estimate,
            "fatigue_level": self.fatigue_accumulated,
            "session_duration_s": time.time() - self.session_start,
            "screen": {"width": self.screen.width, "height": self.screen.height}
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def move_to(x: int, y: int) -> MovementMetrics:
    """Move cursor to position with human-like motion."""
    return AdvancedMouseController().move_to(x, y)


def click_at(x: int, y: int, button: str = "left") -> MovementMetrics:
    """Move to position and click."""
    return AdvancedMouseController().click_at(x, y, button)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Advanced Human-like Mouse Controller v3.0.0")
    print("=" * 70)
    
    controller = AdvancedMouseController()
    
    print(f"\nScreen: {controller.screen.width}x{controller.screen.height}")
    print(f"PID: Kp={controller.pid.kp}, Ki={controller.pid.ki}, Kd={controller.pid.kd}")
    print(f"Human Profile: reaction={controller.profile.reaction_time_mean}ms, "
          f"tremor={controller.profile.tremor_amplitude}px")
    
    try:
        x, y = controller.get_position()
        print(f"Current position: ({x}, {y})")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    
    targets = [
        (960, 540, "center"),
        (150, 150, "top-left"),
        (1750, 150, "top-right"),
        (1750, 900, "bottom-right"),
        (150, 900, "bottom-left"),
        (960, 540, "center"),
    ]
    
    print("\n" + "-" * 70)
    print(f"{'Target':<14} {'Actual':<14} {'Error':>6} {'Time':>8} {'Path':>6} {'Eff':>5} {'Corr':>5}")
    print("-" * 70)
    
    successes = 0
    for tx, ty, name in targets:
        result = controller.move_to(tx, ty)
        
        if result.success:
            successes += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"{status} {name:<12} ({result.actual[0]:4},{result.actual[1]:4}) "
              f"{result.error_euclidean:5.1f}px "
              f"{result.total_duration_ms:6.0f}ms "
              f"{result.path_points:5} "
              f"{result.path_efficiency:4.0%} "
              f"{result.corrections_made:4}")
        
        time.sleep(0.3)
    
    print("-" * 70)
    print(f"\nSuccess: {successes}/{len(targets)}")
    
    stats = controller.get_stats()
    print(f"Learned acceleration: {stats['acceleration_estimate']:.3f}")
    print("=" * 70)
