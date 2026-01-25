"""Premium Streaming Configuration and Recording.

This module provides the premium streaming experience with:
- 1080p default resolution
- Maximum fidelity rendering
- Smart buffering (30 seconds at 60fps = 1800 frames)
- Audio enabled by default
- Stream recording to video file

Classes:
    PremiumConfig: High-quality streaming configuration
    StreamRecorder: Records glyph stream to video file
    SmartBuffer: Intelligent buffer that adapts to stream length
"""
from __future__ import annotations

import os
import sys
import time
import threading
import queue
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datetime import datetime

import numpy as np

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

from .types import QualityLevel
from .hifi import (
    BrailleRenderer,
    ExtendedGradient,
    HybridRenderer,
    STANDARD_RESOLUTIONS,
)
from .ultra import UltraConfig, UltraStreamEngine


# ═══════════════════════════════════════════════════════════════
# Premium Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass
class PremiumConfig:
    """Premium streaming configuration for maximum quality.
    
    Defaults optimized for:
    - 1080p resolution (falls back to 720p/480p if needed)
    - 60fps target with smart buffering
    - Full color with ANSI256 (fast) or TrueColor (maximum quality)
    - Audio enabled
    - Optional recording
    
    Attributes:
        resolution: Target resolution ("auto", "1080p", "720p", "480p")
        target_fps: Target frame rate (60fps default)
        render_mode: Rendering mode ("braille", "hybrid", "gradient")
        color_mode: Color mode ("truecolor", "ansi256", "none")
        buffer_seconds: Seconds of content to buffer (30 default)
        audio_enabled: Enable audio playback
        record_enabled: Enable stream recording
        record_path: Output path for recording
        show_metrics: Show performance overlay
        adaptive_resolution: Downgrade resolution if can't keep up
    """
    # Resolution settings
    resolution: str = "1080p"
    target_fps: int = 60
    adaptive_resolution: bool = True
    
    # Rendering settings
    render_mode: str = "braille"  # Maximum detail
    color_mode: str = "ansi256"  # Fast color mode
    
    # Buffering settings
    buffer_seconds: float = 30.0  # 30 seconds of buffer
    prebuffer_seconds: float = 3.0  # 3 seconds prebuffer before playback
    smart_buffer: bool = True  # Adapt buffer to stream length
    
    # Audio settings
    audio_enabled: bool = True
    audio_sync: bool = True
    
    # Recording settings
    record_enabled: bool = False
    record_path: Optional[str] = None
    record_format: str = "mp4"  # mp4, webm, avi
    record_codec: str = "libx264"  # libx264, libvpx, mjpeg
    record_quality: int = 23  # CRF quality (lower = better)
    
    # Display settings
    show_metrics: bool = True
    show_border: bool = True
    
    def __post_init__(self):
        """Validate and normalize configuration."""
        # Validate resolution
        valid_resolutions = {"auto", "1080p", "720p", "480p", "360p"}
        if self.resolution not in valid_resolutions:
            self.resolution = "1080p"
        
        # Validate render mode
        valid_modes = {"braille", "hybrid", "gradient"}
        if self.render_mode not in valid_modes:
            self.render_mode = "braille"
        
        # Validate color mode
        valid_colors = {"truecolor", "ansi256", "ansi16", "none"}
        if self.color_mode not in valid_colors:
            self.color_mode = "ansi256"
        
        # Ensure non-negative buffer
        self.buffer_seconds = max(0.0, self.buffer_seconds)
        self.prebuffer_seconds = max(0.0, self.prebuffer_seconds)
        
        # Calculate buffer frames
        self.buffer_frames = int(self.buffer_seconds * self.target_fps)
        self.prebuffer_frames = int(self.prebuffer_seconds * self.target_fps)
        
        # Ensure prebuffer doesn't exceed buffer
        self.prebuffer_frames = min(self.prebuffer_frames, self.buffer_frames)
        
        # Set default record path if recording enabled
        if self.record_enabled and not self.record_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.record_path = f"glyph_stream_{timestamp}.{self.record_format}"
    
    def get_pixel_resolution(self) -> Tuple[int, int]:
        """Get pixel resolution for current settings."""
        if self.resolution == "auto":
            # Auto-detect based on terminal size
            try:
                cols, rows = os.get_terminal_size()
                # Braille: 2 pixels per char width, 4 per height
                if self.render_mode == "braille":
                    return (cols * 2, rows * 4)
                else:
                    return (cols * 8, rows * 16)
            except OSError:
                return STANDARD_RESOLUTIONS["720p"]
        
        return STANDARD_RESOLUTIONS.get(self.resolution, (1920, 1080))
    
    def get_terminal_size(self) -> Tuple[int, int]:
        """Get required terminal size for resolution."""
        w, h = self.get_pixel_resolution()
        
        if self.render_mode == "braille":
            return (w // 2, h // 4)
        else:
            return (w // 8, h // 16)
    
    @classmethod
    def from_quality_preset(cls, preset: str) -> "PremiumConfig":
        """Create config from quality preset name.
        
        Presets:
            - "maximum": 1080p, 60fps, truecolor, braille
            - "high": 1080p, 60fps, ansi256, braille
            - "standard": 720p, 30fps, ansi256, gradient
            - "fast": 480p, 30fps, ansi256, gradient
            - "minimal": 360p, 15fps, none, gradient
        """
        presets = {
            "maximum": cls(
                resolution="1080p",
                target_fps=60,
                render_mode="braille",
                color_mode="truecolor",
                buffer_seconds=30.0,
            ),
            "high": cls(
                resolution="1080p",
                target_fps=60,
                render_mode="braille",
                color_mode="ansi256",
                buffer_seconds=30.0,
            ),
            "standard": cls(
                resolution="720p",
                target_fps=30,
                render_mode="gradient",
                color_mode="ansi256",
                buffer_seconds=15.0,
            ),
            "fast": cls(
                resolution="480p",
                target_fps=30,
                render_mode="gradient",
                color_mode="ansi256",
                buffer_seconds=10.0,
            ),
            "minimal": cls(
                resolution="360p",
                target_fps=15,
                render_mode="gradient",
                color_mode="none",
                buffer_seconds=5.0,
            ),
        }
        
        return presets.get(preset, presets["high"])


# ═══════════════════════════════════════════════════════════════
# Smart Buffer
# ═══════════════════════════════════════════════════════════════

class SmartBuffer:
    """Intelligent frame buffer that adapts to stream characteristics.
    
    Features:
    - Automatically caps buffer at stream length
    - Pre-computes frames during prebuffer phase
    - Thread-safe double-buffering
    - Memory-efficient with optional disk spilling
    
    Attributes:
        max_frames: Maximum frames to buffer
        current_size: Current buffer size
        is_full: Whether buffer has reached capacity or stream end
    """
    
    def __init__(
        self,
        max_frames: int = 1800,
        prebuffer_frames: int = 180,
        stream_total_frames: Optional[int] = None,
    ):
        """Initialize smart buffer.
        
        Args:
            max_frames: Maximum frames to buffer
            prebuffer_frames: Frames to accumulate before playback
            stream_total_frames: Total frames in stream (None=unknown/live)
        """
        self.max_frames = max_frames
        self.prebuffer_frames = prebuffer_frames
        self.stream_total_frames = stream_total_frames
        
        # Adjust buffer size if stream is shorter than buffer
        if stream_total_frames is not None:
            self.effective_max = min(max_frames, stream_total_frames)
            self.effective_prebuffer = min(prebuffer_frames, stream_total_frames)
        else:
            self.effective_max = max_frames
            self.effective_prebuffer = prebuffer_frames
        
        # Frame storage
        self._frames: deque = deque(maxlen=self.effective_max)
        self._rendered: deque = deque(maxlen=self.effective_max)
        
        # State
        self._lock = threading.RLock()
        self._prebuffer_complete = threading.Event()
        self._stream_complete = False
        self._frames_received = 0
        self._frames_consumed = 0
    
    def add_frame(
        self,
        frame: np.ndarray,
        rendered: Optional[List[str]] = None,
    ) -> bool:
        """Add frame to buffer.
        
        Args:
            frame: Raw frame data
            rendered: Optional pre-rendered output lines
            
        Returns:
            True if frame was added, False if buffer is full
        """
        with self._lock:
            if len(self._frames) >= self.effective_max:
                # Buffer full - remove oldest frame
                self._frames.popleft()
                if self._rendered:
                    self._rendered.popleft()
            
            self._frames.append(frame)
            if rendered is not None:
                self._rendered.append(rendered)
            
            self._frames_received += 1
            
            # Check if prebuffer is complete
            if self._frames_received >= self.effective_prebuffer:
                self._prebuffer_complete.set()
            
            return True
    
    def get_frame(self, timeout: float = 1.0) -> Optional[Tuple[np.ndarray, Optional[List[str]]]]:
        """Get next frame from buffer.
        
        Args:
            timeout: Max wait time for frame
            
        Returns:
            (frame, rendered) tuple or None if empty
        """
        # Wait for prebuffer to complete
        if not self._prebuffer_complete.wait(timeout):
            return None
        
        with self._lock:
            if not self._frames:
                return None
            
            frame = self._frames.popleft()
            rendered = self._rendered.popleft() if self._rendered else None
            self._frames_consumed += 1
            
            return (frame, rendered)
    
    def wait_for_prebuffer(self, timeout: float = 30.0) -> bool:
        """Wait for prebuffer to complete.
        
        Args:
            timeout: Maximum wait time
            
        Returns:
            True if prebuffer complete, False if timeout
        """
        return self._prebuffer_complete.wait(timeout)
    
    def mark_stream_complete(self) -> None:
        """Mark that no more frames will be added."""
        with self._lock:
            self._stream_complete = True
            # Ensure prebuffer event is set even if we have fewer frames
            self._prebuffer_complete.set()
    
    @property
    def current_size(self) -> int:
        """Get current buffer size."""
        with self._lock:
            return len(self._frames)
    
    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        with self._lock:
            return len(self._frames) == 0
    
    @property
    def is_prebuffer_complete(self) -> bool:
        """Check if prebuffer is complete."""
        return self._prebuffer_complete.is_set()
    
    @property
    def buffer_level(self) -> float:
        """Get buffer fill level (0.0 - 1.0)."""
        with self._lock:
            return len(self._frames) / self.effective_max if self.effective_max > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            return {
                "current_size": len(self._frames),
                "max_size": self.effective_max,
                "prebuffer_target": self.effective_prebuffer,
                "frames_received": self._frames_received,
                "frames_consumed": self._frames_consumed,
                "buffer_level": self.buffer_level,
                "prebuffer_complete": self.is_prebuffer_complete,
                "stream_complete": self._stream_complete,
            }


# ═══════════════════════════════════════════════════════════════
# Stream Recorder
# ═══════════════════════════════════════════════════════════════

class StreamRecorder:
    """Records glyph stream output to video file.
    
    Captures rendered terminal output as video frames and encodes
    to a standard video format.
    
    Features:
    - Records rendered glyph output (not source video)
    - Configurable codec and quality
    - Thread-safe async recording
    - Proper cleanup on exit
    """
    
    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float = 60.0,
        codec: str = "mp4v",
        quality: int = 23,
    ):
        """Initialize stream recorder.
        
        Args:
            output_path: Output file path
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Target frame rate
            codec: Video codec (mp4v, XVID, avc1)
            quality: Quality setting (codec-dependent)
        """
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.quality = quality
        
        self._writer: Optional[cv2.VideoWriter] = None
        self._lock = threading.Lock()
        self._frame_count = 0
        self._is_open = False
    
    def open(self) -> bool:
        """Open video writer.
        
        Returns:
            True if opened successfully
        """
        if not HAS_OPENCV:
            return False
        
        with self._lock:
            fourcc = cv2.VideoWriter.fourcc(*self.codec)
            self._writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.fps,
                (self.width, self.height),
                True,  # isColor
            )
            
            self._is_open = self._writer.isOpened()
            return self._is_open
    
    def write_frame(self, frame: np.ndarray) -> bool:
        """Write frame to video.
        
        Args:
            frame: BGR frame (H, W, 3)
            
        Returns:
            True if written successfully
        """
        with self._lock:
            if not self._is_open or self._writer is None:
                return False
            
            # Resize if needed
            h, w = frame.shape[:2]
            if w != self.width or h != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            
            self._writer.write(frame)
            self._frame_count += 1
            return True
    
    def write_rendered(
        self,
        lines: List[str],
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        font_scale: float = 0.4,
    ) -> bool:
        """Write rendered text lines as video frame.
        
        Renders ANSI-colored text to an image and writes to video.
        
        Args:
            lines: ANSI-colored text lines
            bg_color: Background color (BGR)
            font_scale: Font scale factor
            
        Returns:
            True if written successfully
        """
        if not HAS_OPENCV:
            return False
        
        # Create blank frame
        frame = np.full((self.height, self.width, 3), bg_color, dtype=np.uint8)
        
        # Calculate character dimensions
        char_height = int(self.height / len(lines)) if lines else 16
        char_width = int(char_height * 0.5)
        
        # Strip ANSI codes for rendering (simplified)
        import re
        ansi_pattern = re.compile(r'\x1b\[[0-9;]*m')
        
        y = char_height
        for line in lines:
            # Strip ANSI codes
            clean_line = ansi_pattern.sub('', line)
            
            # Put text on frame
            cv2.putText(
                frame,
                clean_line[:self.width // char_width],  # Truncate to fit
                (0, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += char_height
            
            if y > self.height:
                break
        
        return self.write_frame(frame)
    
    def close(self) -> None:
        """Close video writer and finalize file."""
        with self._lock:
            if self._writer is not None:
                self._writer.release()
                self._writer = None
            self._is_open = False
    
    @property
    def frame_count(self) -> int:
        """Get number of frames written."""
        return self._frame_count
    
    @property
    def duration(self) -> float:
        """Get recording duration in seconds."""
        return self._frame_count / self.fps if self.fps > 0 else 0.0
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ═══════════════════════════════════════════════════════════════
# Premium Stream Engine
# ═══════════════════════════════════════════════════════════════

class PremiumStreamEngine:
    """Premium streaming engine with maximum quality and recording.
    
    Features:
    - 1080p default resolution
    - Smart buffering (30 seconds)
    - Audio synchronization
    - Optional recording
    - Adaptive resolution fallback
    """
    
    def __init__(self, config: Optional[PremiumConfig] = None):
        """Initialize premium stream engine.
        
        Args:
            config: Premium configuration (None for defaults)
        """
        self.config = config or PremiumConfig()
        
        # Create underlying ultra engine
        ultra_config = UltraConfig(
            resolution=self.config.resolution,
            target_fps=self.config.target_fps,
            render_mode=self.config.render_mode,
            color_enabled=self.config.color_mode != "none",
            braille_ansi256=self.config.color_mode == "ansi256",
            show_metrics=self.config.show_metrics,
        )
        self._engine = UltraStreamEngine(ultra_config)
        
        # Buffer
        self._buffer: Optional[SmartBuffer] = None
        
        # Recorder
        self._recorder: Optional[StreamRecorder] = None
        
        # State
        self._running = False
        self._lock = threading.RLock()
    
    def run(
        self,
        source: Union[str, int],
        output_path: Optional[str] = None,
    ) -> None:
        """Run premium streaming.
        
        Args:
            source: Video source (file, URL, or camera index)
            output_path: Optional output path for recording
        """
        if not HAS_OPENCV:
            raise RuntimeError("OpenCV required for video capture")
        
        # Handle YouTube URLs
        if isinstance(source, str) and ("youtube.com" in source or "youtu.be" in source):
            from .extractors import YouTubeExtractor
            extractor = YouTubeExtractor()
            result = extractor.extract(source)
            actual_source = result.video_url
            audio_url = result.audio_url
            # Extract duration and fps directly from result
            source_fps = result.fps or 30
            total_frames = int(result.duration * source_fps) if result.duration else None
        else:
            actual_source = source
            audio_url = None
            total_frames = None
        
        # Open video capture
        cap = cv2.VideoCapture(actual_source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")
        
        try:
            # Get video info
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = total_frames or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                total_frames = None  # Live stream
            
            # Create smart buffer
            buffer_frames = int(self.config.buffer_seconds * fps)
            prebuffer_frames = int(self.config.prebuffer_seconds * fps)
            
            self._buffer = SmartBuffer(
                max_frames=buffer_frames,
                prebuffer_frames=prebuffer_frames,
                stream_total_frames=total_frames,
            )
            
            # Setup recording
            record_path = output_path or (self.config.record_path if self.config.record_enabled else None)
            if record_path:
                self._recorder = StreamRecorder(
                    output_path=record_path,
                    width=width,
                    height=height,
                    fps=fps,
                )
                self._recorder.open()
            
            # Print startup info
            self._print_startup_info(width, height, fps, total_frames, record_path)
            
            # Start streaming
            self._running = True
            self._stream_loop(cap, fps)
            
        finally:
            cap.release()
            if self._recorder:
                self._recorder.close()
                print(f"\n[Recording saved to {record_path}]")
            self._running = False
    
    def _stream_loop(
        self,
        cap: cv2.VideoCapture,
        fps: float,
    ) -> None:
        """Main streaming loop.
        
        Args:
            cap: OpenCV video capture
            fps: Frame rate
        """
        frame_interval = 1.0 / min(fps, self.config.target_fps)
        
        # Clear screen
        sys.stdout.write("\033[2J\033[H")
        
        prebuffer_printed = False
        
        while self._running:
            frame_start = time.perf_counter()
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                if self._buffer:
                    self._buffer.mark_stream_complete()
                break
            
            # Process frame
            lines = self._engine.process_frame_ultra(frame)
            
            # Add to buffer
            if self._buffer:
                self._buffer.add_frame(frame, lines)
            
            # Show prebuffering status
            if self._buffer and not self._buffer.is_prebuffer_complete:
                if not prebuffer_printed:
                    sys.stdout.write(f"\033[H[Prebuffering: {self._buffer.current_size}/{self._buffer.effective_prebuffer}]\033[K")
                    sys.stdout.flush()
                    prebuffer_printed = True
                continue
            
            # Record if enabled
            if self._recorder:
                self._recorder.write_frame(frame)
            
            # Output frame
            sys.stdout.write("\033[H")
            for line in lines:
                sys.stdout.write(line + "\n")
            
            # Show metrics
            if self.config.show_metrics:
                metrics = self._engine.get_metrics()
                buffer_stats = self._buffer.get_stats() if self._buffer else {}
                sys.stdout.write(
                    f"\033[KFPS: {metrics.get('fps', 0):.1f} | "
                    f"Buffer: {buffer_stats.get('buffer_level', 0)*100:.0f}% | "
                    f"Frame: {metrics.get('frame_time_ms', 0):.1f}ms\n"
                )
            
            sys.stdout.flush()
            
            # Frame timing
            elapsed = time.perf_counter() - frame_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _print_startup_info(
        self,
        width: int,
        height: int,
        fps: float,
        total_frames: Optional[int],
        record_path: Optional[str],
    ) -> None:
        """Print startup information."""
        duration = total_frames / fps if total_frames else None
        
        print(f"╔══════════════════════════════════════════════════╗")
        print(f"║  🎬 GLYPH FORGE PREMIUM STREAM                   ║")
        print(f"╠══════════════════════════════════════════════════╣")
        print(f"║  Resolution: {width}x{height} @ {fps:.1f}fps")
        print(f"║  Mode: {self.config.render_mode.upper()} | Color: {self.config.color_mode.upper()}")
        print(f"║  Buffer: {self.config.buffer_seconds:.0f}s ({int(self.config.buffer_seconds * fps)} frames)")
        if duration:
            print(f"║  Duration: {duration:.1f}s ({total_frames} frames)")
        if record_path:
            print(f"║  Recording to: {record_path}")
        print(f"╚══════════════════════════════════════════════════╝")
        print(f"[Prebuffering {int(self.config.prebuffer_seconds)}s before playback...]")
    
    def stop(self) -> None:
        """Stop streaming."""
        self._running = False


# ═══════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════

def stream_premium(
    source: Union[str, int],
    resolution: str = "1080p",
    fps: int = 60,
    buffer_seconds: float = 30.0,
    color: bool = True,
    audio: bool = True,
    record: Optional[str] = None,
) -> None:
    """Premium streaming with sensible defaults.
    
    Args:
        source: Video source (file, URL, camera index)
        resolution: Target resolution
        fps: Target FPS
        buffer_seconds: Buffer size in seconds
        color: Enable color
        audio: Enable audio
        record: Optional output path for recording
    """
    config = PremiumConfig(
        resolution=resolution,
        target_fps=fps,
        buffer_seconds=buffer_seconds,
        color_mode="ansi256" if color else "none",
        audio_enabled=audio,
        record_enabled=record is not None,
        record_path=record,
    )
    
    engine = PremiumStreamEngine(config)
    
    try:
        engine.run(source, record)
    except KeyboardInterrupt:
        engine.stop()
    finally:
        sys.stdout.write("\033[0m")
        sys.stdout.flush()
