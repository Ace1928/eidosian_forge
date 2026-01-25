"""Core type definitions for the Glyph Stream system.

This module provides all core types, enums, dataclasses and TypedDicts
used throughout the streaming system.

Types:
    EdgeDetector: Edge detection algorithm enum
    GradientResult: Edge detection tensor container
    QualityLevel: Rendering quality presets
    VideoInfo: Normalized video metadata
    PerformanceStats: Performance metrics container
    TextStyle: Text rendering style presets
    RenderThresholds: Adaptive quality thresholds
    StreamMetrics: High-precision performance metrics
    RenderParameters: Adaptive rendering parameters
"""
from __future__ import annotations

import collections
import math
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from typing import Any, Dict, Optional, Tuple, TypedDict

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════
# Edge Detection Types
# ═══════════════════════════════════════════════════════════════

class EdgeDetector(Enum):
    """Edge detection algorithms with specialized dimensional sensitivities.
    
    Each algorithm offers precise advantages for specific visual contexts,
    with dynamic parameter optimization based on content characteristics.
    
    Attributes:
        SOBEL: Balanced gradient sensitivity with directional precision
        PREWITT: Enhanced noise immunity for high-contrast boundaries
        SCHARR: Superior diagonal detection with rotational invariance
        LAPLACIAN: Omnidirectional boundary detection
        CANNY: Maximum edge coherence through dual-threshold hysteresis
    """
    SOBEL = auto()     # Standard workhorse with balanced gradients
    PREWITT = auto()   # Signal-to-noise optimization
    SCHARR = auto()    # Angular precision for diagonal structures
    LAPLACIAN = auto() # Second-order differential for all directions
    CANNY = auto()     # Surgical edge detection


class GradientResult(TypedDict):
    """Edge detection tensor container with vectorized components.
    
    Provides comprehensive gradient analysis with normalized magnitude
    and directional components for advanced dimensional rendering.
    
    Attributes:
        magnitude: Normalized edge intensity tensor (0-255, uint8)
        gradient_x: Horizontal gradient component tensor (float32)
        gradient_y: Vertical gradient component tensor (float32)
        direction: Angular orientation tensor in radians (float32)
    """
    magnitude: np.ndarray[Any, Any]
    gradient_x: np.ndarray[Any, Any]
    gradient_y: np.ndarray[Any, Any]
    direction: np.ndarray[Any, Any]


# ═══════════════════════════════════════════════════════════════
# Quality and Style Types
# ═══════════════════════════════════════════════════════════════

class QualityLevel(IntEnum):
    """Rendering quality presets with adaptive performance scaling.
    
    Provides standardized quality tiers for real-time adaptation
    based on system capabilities and performance metrics.
    
    Attributes:
        MINIMAL: Maximum performance for constrained systems (0)
        LOW: Reduced quality with optimized resources (1)
        STANDARD: Balanced quality/performance (2)
        HIGH: Enhanced detail with selective improvements (3)
        MAXIMUM: Maximum fidelity with all enhancements (4)
    """
    MINIMAL = 0
    LOW = 1
    STANDARD = 2
    HIGH = 3
    MAXIMUM = 4


class TextStyle(Enum):
    """Text rendering style presets with aesthetic parameters.
    
    Defines rendering configurations with complete formatting
    specifications for diverse visual presentation needs.
    
    Attributes:
        SIMPLE: Clean minimal presentation
        STYLED: Enhanced typography with borders
        RAINBOW: Multi-color spectral gradient
        RANDOM: Dynamic procedural styling
    """
    SIMPLE = auto()
    STYLED = auto()
    RAINBOW = auto()
    RANDOM = auto()


# ═══════════════════════════════════════════════════════════════
# Video and Performance Types
# ═══════════════════════════════════════════════════════════════

class VideoInfo:
    """Normalized video metadata container with validated fields.
    
    Maintains structurally perfect video source information with
    comprehensive validation and type safety.
    
    Attributes:
        url: Direct stream address for network sources
        title: Content identifier with proper sanitization
        duration: Total playback seconds (None if streaming)
        format: Content format identifier (codec/container)
        width: Frame width in pixels (None if unknown)
        height: Frame height in pixels (None if unknown)
        fps: Frames per second (None if variable)
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        title: str = "Unknown",
        duration: Optional[int] = None,
        format: str = "unknown",
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
    ):
        self.url = url
        self.title = title
        self.duration = duration
        self.format = format
        self.width = width
        self.height = height
        self.fps = fps

    @classmethod
    def from_capture(
        cls, capture: Any, source_name: str, stream_format: str
    ) -> "VideoInfo":
        """Extract validated metadata from capture device.
        
        Args:
            capture: OpenCV VideoCapture object
            source_name: Descriptive content identifier
            stream_format: Format classification ('file', 'youtube', etc.)
        
        Returns:
            VideoInfo: Normalized metadata with validated fields
        """
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Validate with domain-specific constraints
            width = width if width > 0 else None
            height = height if height > 0 else None
            fps = fps if 0 < fps < 1000 else None
            duration = int(frame_count / fps) if frame_count > 0 and fps else None
            
            # Extract clean title from path/URL
            title = (
                os.path.basename(source_name)
                if os.path.exists(source_name)
                else source_name
            )
            if len(title) > 40:
                title = title[:37] + "..."
            
            return cls(
                title=title,
                format=stream_format,
                width=width,
                height=height,
                fps=fps,
                duration=duration,
            )
        except Exception:
            return cls(title=str(source_name), format=stream_format)


class PerformanceStats(TypedDict):
    """Immutable performance metrics with normalized analysis fields.
    
    Contains comprehensive performance data with statistical validity
    guarantees for real-time quality adaptation.
    
    Attributes:
        avg_render_time: Mean rendering time in milliseconds
        avg_fps: Mean frames per second across sample window
        effective_fps: True delivered framerate across entire runtime
        total_frames: Cumulative successfully processed frames
        dropped_frames: Frames abandoned due to timing constraints
        drop_ratio: Proportion of dropped frames (0.0-1.0)
        stability: Timing consistency coefficient (0.0-1.0)
    """
    avg_render_time: float
    avg_fps: float
    effective_fps: float
    total_frames: int
    dropped_frames: int
    drop_ratio: float
    stability: float


# ═══════════════════════════════════════════════════════════════
# Rendering Configuration Types
# ═══════════════════════════════════════════════════════════════

@dataclass
class RenderThresholds:
    """Adaptive quality thresholds based on render time performance.
    
    Attributes:
        reduce_ms: Threshold above which quality should be reduced
        improve_ms: Threshold below which quality can be improved
    """
    reduce_ms: float
    improve_ms: float
    
    @classmethod
    def from_target_fps(
        cls,
        target_fps: float,
        reduce_ratio: float = 0.9,
        improve_ratio: float = 0.6,
    ) -> "RenderThresholds":
        """Create optimal thresholds from target FPS.
        
        Args:
            target_fps: Desired frames per second
            reduce_ratio: Percentage triggering quality reduction
            improve_ratio: Percentage allowing quality improvement
            
        Returns:
            RenderThresholds: Calculated thresholds
        """
        frame_budget_ms = 1000.0 / target_fps
        return cls(
            reduce_ms=frame_budget_ms * reduce_ratio,
            improve_ms=frame_budget_ms * improve_ratio,
        )


class StreamMetrics:
    """High-precision performance metrics with statistical analysis.
    
    Thread-safe performance tracking for realtime visualization with
    rolling statistics and comprehensive analytics.
    
    Attributes:
        frames_processed: Total successfully rendered frames
        current_fps: Most recently calculated frames per second
        average_render_time: Mean render time in milliseconds
    """
    
    def __init__(self, sample_size: int = 30) -> None:
        """Initialize performance tracking.
        
        Args:
            sample_size: Maximum samples for rolling statistics
        """
        self._lock = threading.RLock()
        self.frames_processed: int = 0
        self.dropped_frames: int = 0
        self.start_time: float = time.time()
        self.current_fps: float = 0.0
        
        self._render_times: collections.deque[float] = collections.deque(
            maxlen=sample_size
        )
        self._fps_samples: collections.deque[float] = collections.deque(
            maxlen=sample_size
        )
        self._last_fps_time = time.time()
        self._frames_since_fps_update = 0
        
    def update_fps(self) -> float:
        """Calculate current FPS with time-weighted accuracy.
        
        Returns:
            float: Current frames per second
        """
        with self._lock:
            now = time.time()
            elapsed = now - self._last_fps_time
            
            if elapsed >= 1.0 and self._frames_since_fps_update > 0:
                self.current_fps = self._frames_since_fps_update / elapsed
                self._fps_samples.append(self.current_fps)
                self._last_fps_time = now
                self._frames_since_fps_update = 0
                
            return self.current_fps
        
    def record_render(self, duration: float) -> None:
        """Record frame rendering duration.
        
        Args:
            duration: Render time in seconds
        """
        with self._lock:
            self._render_times.append(duration * 1000)
        
    def record_frame(self) -> None:
        """Record successful frame processing."""
        with self._lock:
            self.frames_processed += 1
            self._frames_since_fps_update += 1
        
    def record_dropped(self) -> None:
        """Record frame drop for analytics."""
        with self._lock:
            self.dropped_frames += 1
    
    @property
    def average_render_time(self) -> float:
        """Calculate mean render time in milliseconds."""
        with self._lock:
            return (
                sum(self._render_times) / len(self._render_times)
                if self._render_times
                else 0.0
            )
    
    @property
    def effective_fps(self) -> float:
        """Calculate overall FPS across entire runtime."""
        with self._lock:
            duration = time.time() - self.start_time
            return self.frames_processed / max(0.001, duration)
    
    @property
    def drop_ratio(self) -> float:
        """Calculate percentage of frames dropped (0.0-1.0)."""
        with self._lock:
            total = self.frames_processed + self.dropped_frames
            return self.dropped_frames / max(1, total)
    
    def get_stats(self) -> PerformanceStats:
        """Generate comprehensive performance analytics.
        
        Returns:
            PerformanceStats: Complete metrics dictionary
        """
        with self._lock:
            avg_fps = (
                sum(self._fps_samples) / len(self._fps_samples)
                if self._fps_samples
                else 0.0
            )
            
            stability = 1.0
            if len(self._render_times) > 1:
                try:
                    mean = self.average_render_time
                    if mean > 0:
                        variance = sum(
                            (t - mean) ** 2 for t in self._render_times
                        ) / len(self._render_times)
                        cv = min(1.0, math.sqrt(variance) / mean)
                        stability = 1.0 - cv
                except (ZeroDivisionError, OverflowError):
                    pass
            
            return {
                "avg_render_time": self.average_render_time,
                "avg_fps": avg_fps,
                "effective_fps": self.effective_fps,
                "total_frames": self.frames_processed,
                "dropped_frames": self.dropped_frames,
                "drop_ratio": self.drop_ratio,
                "stability": stability,
            }


class RenderParameters:
    """Adaptive rendering parameters with quality management.
    
    Provides intelligent quality scaling across multiple dimensions
    with optimized parameter sets for different performance targets.
    
    Attributes:
        scale: Detail enhancement factor (1-4)
        width: Block width in pixels
        height: Block height in pixels
        threshold: Edge detection sensitivity (0-100)
        quality_level: Current quality preset (MINIMAL-MAXIMUM)
    """
    
    def __init__(
        self,
        scale: int = 1,
        width: int = 2,
        height: int = 4,
        threshold: int = 50,
        optimal_width: int = 120,
        optimal_height: int = 40,
        quality_level: QualityLevel = QualityLevel.STANDARD,
    ) -> None:
        """Initialize with balanced default parameters.
        
        Args:
            scale: Detail enhancement factor
            width: Character cell width
            height: Character cell height
            threshold: Edge sensitivity
            optimal_width: Reference width for quality scaling
            optimal_height: Reference height for quality scaling
            quality_level: Initial quality preset
        """
        self._lock = threading.RLock()
        self.scale = scale
        self.width = width
        self.height = height
        self.threshold = threshold
        self.optimal_width = optimal_width
        self.optimal_height = optimal_height
        self.quality_level = quality_level
        
        # Quality presets
        self._quality_configs: Dict[QualityLevel, Dict[str, int]] = {
            QualityLevel.MINIMAL: {"scale": 1, "width": 3, "height": 6},
            QualityLevel.LOW: {"scale": 1, "width": 2, "height": 5},
            QualityLevel.STANDARD: {"scale": 1, "width": 2, "height": 4},
            QualityLevel.HIGH: {"scale": 2, "width": 2, "height": 4},
            QualityLevel.MAXIMUM: {"scale": 4, "width": 1, "height": 2},
        }
    
    def adjust_quality(self, direction: int) -> bool:
        """Adjust quality level by direction.
        
        Args:
            direction: +1 to increase, -1 to decrease
            
        Returns:
            bool: True if quality was adjusted
        """
        with self._lock:
            new_level = self.quality_level + direction
            if QualityLevel.MINIMAL <= new_level <= QualityLevel.MAXIMUM:
                self.quality_level = QualityLevel(new_level)
                config = self._quality_configs[self.quality_level]
                self.scale = config["scale"]
                self.width = config["width"]
                self.height = config["height"]
                return True
            return False
    
    def increase_quality(self) -> bool:
        """Increase rendering quality if possible."""
        return self.adjust_quality(1)
    
    def decrease_quality(self) -> bool:
        """Decrease rendering quality if possible."""
        return self.adjust_quality(-1)
    
    def get_effective_dimensions(
        self, frame_width: int, frame_height: int
    ) -> Tuple[int, int]:
        """Calculate effective output dimensions.
        
        Args:
            frame_width: Source frame width
            frame_height: Source frame height
            
        Returns:
            Tuple[int, int]: (output_width, output_height) in characters
        """
        with self._lock:
            effective_scale = self.scale
            char_width = max(1, frame_width // self.width // effective_scale)
            char_height = max(1, frame_height // self.height // effective_scale)
            return (
                min(char_width, self.optimal_width),
                min(char_height, self.optimal_height),
            )
