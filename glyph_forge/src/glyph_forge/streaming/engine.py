"""High-performance streaming engine.

This module provides the main StreamEngine class that orchestrates
video capture, frame processing, rendering, buffering, and audio playback
for high-performance terminal streaming.

Classes:
    StreamEngine: Main streaming orchestrator
    StreamConfig: Configuration for streaming
"""
from __future__ import annotations

import os
import sys
import time
import threading
import signal
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

from .types import (
    QualityLevel,
    VideoInfo,
    StreamMetrics,
    RenderParameters,
    RenderThresholds,
)
from .extractors import (
    VideoSourceExtractor,
    YouTubeExtractor,
    ExtractionResult,
    StreamExtractionError,
    DependencyError,
)
from .processors import FrameProcessor
from .renderers import CharacterRenderer, FrameRenderer, FrameBuffer
from .audio import AudioPlayer, AudioSync


# ═══════════════════════════════════════════════════════════════
# Stream Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass
class StreamConfig:
    """Configuration for stream processing.
    
    All settings for controlling stream behavior, quality,
    and display options.
    """
    # Quality settings
    scale_factor: int = 1
    block_width: int = 2
    block_height: int = 4
    quality_level: QualityLevel = QualityLevel.STANDARD
    
    # Processing settings
    edge_threshold: int = 50
    algorithm: str = "sobel"
    gradient: str = "standard"
    
    # Display settings
    color_enabled: bool = True
    show_edges: bool = True
    show_border: bool = True
    show_stats: bool = True
    
    # Playback settings
    target_fps: Optional[float] = None  # None = match source
    audio_enabled: bool = True
    adaptive_quality: bool = False  # Disabled by default
    
    # Buffering settings
    buffer_size: int = 60  # Frames
    prebuffer_frames: int = 30  # Frames to buffer before playback
    prebuffer_timeout: float = 10.0  # Max wait for prebuffer
    
    # Performance settings
    max_workers: int = 4
    frame_skip_threshold: float = 0.1  # Skip frames if behind by this much
    
    # Terminal settings
    terminal_width: Optional[int] = None  # None = auto-detect
    terminal_height: Optional[int] = None  # None = auto-detect
    
    def __post_init__(self):
        """Validate and normalize configuration."""
        self.scale_factor = max(1, min(4, self.scale_factor))
        self.block_width = max(1, min(8, self.block_width))
        self.block_height = max(1, min(16, self.block_height))
        self.edge_threshold = max(0, min(255, self.edge_threshold))
        self.buffer_size = max(10, min(300, self.buffer_size))
        self.prebuffer_frames = max(0, min(self.buffer_size, self.prebuffer_frames))


# ═══════════════════════════════════════════════════════════════
# Stream Engine
# ═══════════════════════════════════════════════════════════════

class StreamEngine:
    """High-performance streaming engine.
    
    Orchestrates the complete pipeline:
    - Video capture (files, webcams, YouTube, network streams)
    - Frame processing (edge detection, color extraction)
    - Character rendering (glyph mapping, ANSI colors)
    - Buffered playback (frame timing, sync)
    - Audio playback (synchronized with video)
    
    Features:
    - Pre-buffering for smooth playback
    - Optional adaptive quality adjustment
    - Audio synchronization
    - Thread-safe operation
    - Graceful shutdown
    
    Attributes:
        config: Stream configuration
        metrics: Performance metrics
        is_running: Whether streaming is active
    """
    
    def __init__(self, config: Optional[StreamConfig] = None):
        """Initialize stream engine.
        
        Args:
            config: Stream configuration (None for defaults)
        """
        self.config = config or StreamConfig()
        self.metrics = StreamMetrics()
        
        # State
        self._running = False
        self._paused = False
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        
        # Components (initialized on run)
        self._capture: Optional[cv2.VideoCapture] = None
        self._processor: Optional[FrameProcessor] = None
        self._char_renderer: Optional[CharacterRenderer] = None
        self._frame_renderer: Optional[FrameRenderer] = None
        self._frame_buffer: Optional[FrameBuffer] = None
        self._audio_player: Optional[AudioPlayer] = None
        self._audio_sync: Optional[AudioSync] = None
        
        # Thread pool for parallel processing
        self._executor: Optional[ThreadPoolExecutor] = None
        
        # Render parameters (for adaptive quality)
        self._render_params: Optional[RenderParameters] = None
        self._quality_thresholds: Optional[RenderThresholds] = None
        
        # Source info
        self._source_info: Optional[ExtractionResult] = None
        self._source_fps: float = 30.0
        
        # Callbacks
        self._on_frame: Optional[Callable[[List[str]], None]] = None
        self._on_error: Optional[Callable[[Exception], None]] = None
    
    @property
    def is_running(self) -> bool:
        """Check if streaming is active."""
        return self._running
    
    @property
    def is_paused(self) -> bool:
        """Check if streaming is paused."""
        return self._paused
    
    def run(
        self,
        source: Union[str, int, Path],
        on_frame: Optional[Callable[[List[str]], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        """Start streaming from source.
        
        This is the main entry point. Blocks until streaming ends.
        
        Args:
            source: Video source (URL, file path, webcam index)
            on_frame: Callback for each rendered frame
            on_error: Callback for errors
        """
        if not HAS_OPENCV:
            raise DependencyError(
                "opencv-python",
                "pip install opencv-python",
                "video streaming",
            )
        
        self._on_frame = on_frame
        self._on_error = on_error
        
        try:
            self._setup(source)
            self._stream_loop()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            if self._on_error:
                self._on_error(e)
            else:
                raise
        finally:
            self._cleanup()
    
    def run_async(
        self,
        source: Union[str, int, Path],
        on_frame: Optional[Callable[[List[str]], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> threading.Thread:
        """Start streaming in a background thread.
        
        Args:
            source: Video source
            on_frame: Callback for each rendered frame
            on_error: Callback for errors
            
        Returns:
            The background thread
        """
        thread = threading.Thread(
            target=self.run,
            args=(source, on_frame, on_error),
            daemon=True,
        )
        thread.start()
        return thread
    
    def stop(self) -> None:
        """Stop streaming."""
        self._stop_event.set()
        self._running = False
    
    def pause(self) -> None:
        """Pause streaming."""
        with self._lock:
            self._paused = True
            if self._audio_player:
                self._audio_player.pause()
    
    def resume(self) -> None:
        """Resume streaming."""
        with self._lock:
            self._paused = False
            if self._audio_player:
                self._audio_player.resume()
    
    def toggle_pause(self) -> bool:
        """Toggle pause state.
        
        Returns:
            New pause state
        """
        with self._lock:
            if self._paused:
                self.resume()
            else:
                self.pause()
            return self._paused
    
    def _setup(self, source: Union[str, int, Path]) -> None:
        """Set up streaming components.
        
        Args:
            source: Video source
        """
        # Get terminal dimensions
        term_width = self.config.terminal_width
        term_height = self.config.terminal_height
        
        if term_width is None or term_height is None:
            try:
                size = os.get_terminal_size()
                term_width = term_width or size.columns
                term_height = term_height or size.lines
            except Exception:
                term_width = term_width or 120
                term_height = term_height or 40
        
        # Extract source info
        print(f"🔍 Extracting source info...", end="", flush=True)
        self._source_info = VideoSourceExtractor.extract(
            source,
            include_audio=self.config.audio_enabled,
        )
        print(f"\r✓ Source: {self._source_info.title}                    ")
        
        # Determine target FPS
        if self.config.target_fps:
            self._source_fps = self.config.target_fps
        elif self._source_info.fps:
            self._source_fps = self._source_info.fps
        else:
            self._source_fps = 30.0
        
        # Open video capture
        video_source = self._source_info.video_url
        if isinstance(source, int):
            video_source = source
        
        self._capture = cv2.VideoCapture(video_source)
        if not self._capture.isOpened():
            raise StreamExtractionError(
                f"Cannot open video source: {source}",
                category="capture_error",
            )
        
        # Initialize components
        self._processor = FrameProcessor(
            scale_factor=self.config.scale_factor,
            block_width=self.config.block_width,
            block_height=self.config.block_height,
            edge_threshold=self.config.edge_threshold,
            algorithm=self.config.algorithm,
            color_enabled=self.config.color_enabled,
            max_workers=self.config.max_workers,
        )
        
        self._char_renderer = CharacterRenderer(
            gradient=self.config.gradient,
            use_unicode=True,
            use_color=self.config.color_enabled,
            edge_mode="enhanced" if self.config.show_edges else "none",
            edge_threshold=self.config.edge_threshold // 3,
        )
        
        self._frame_renderer = FrameRenderer(
            terminal_width=term_width,
            terminal_height=term_height,
            use_unicode=True,
            show_border=self.config.show_border,
            show_stats=self.config.show_stats,
        )
        
        self._frame_buffer = FrameBuffer(
            capacity=self.config.buffer_size,
            target_fps=self._source_fps,
        )
        
        # Render parameters for adaptive quality
        self._render_params = RenderParameters(
            scale=self.config.scale_factor,
            width=self.config.block_width,
            height=self.config.block_height,
            threshold=self.config.edge_threshold,
            optimal_width=term_width - 4,
            optimal_height=term_height - 6,
            quality_level=self.config.quality_level,
        )
        
        if self.config.adaptive_quality:
            self._quality_thresholds = RenderThresholds.from_target_fps(
                self._source_fps
            )
        
        # Set up audio if available
        if (self.config.audio_enabled and 
            self._source_info.has_audio and
            not self._source_info.is_live):
            
            self._audio_player = AudioPlayer()
            if self._audio_player.is_available:
                if self._audio_player.load(self._source_info.audio_url):
                    self._audio_sync = AudioSync(
                        audio_player=self._audio_player,
                        target_fps=self._source_fps,
                    )
        
        # Initialize executor
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.max_workers + 2,  # +2 for capture and display
            thread_name_prefix="stream",
        )
        
        self._running = True
        self._stop_event.clear()
        self.metrics = StreamMetrics()
    
    def _stream_loop(self) -> None:
        """Main streaming loop with buffering."""
        # Calculate target dimensions
        term_width = self._frame_renderer.terminal_width - 4
        term_height = self._frame_renderer.terminal_height - 6
        
        # Pre-buffer frames
        if self.config.prebuffer_frames > 0:
            self._prebuffer(term_width, term_height)
        
        # Start audio playback
        if self._audio_player and self._audio_sync:
            self._audio_player.play()
            self._audio_sync.start()
        
        # Display loop
        frame_duration = 1.0 / self._source_fps
        start_time = time.time()
        frame_count = 0
        
        # Clear screen
        self._clear_screen()
        
        # Start capture thread
        capture_thread = threading.Thread(
            target=self._capture_loop,
            args=(term_width, term_height),
            daemon=True,
        )
        capture_thread.start()
        
        try:
            while self._running and not self._stop_event.is_set():
                # Handle pause
                if self._paused:
                    time.sleep(0.1)
                    continue
                
                # Get frame from buffer
                frame_data = self._frame_buffer.get(block=True, timeout=1.0)
                
                if frame_data is None:
                    # Buffer empty, check if capture is done
                    if not capture_thread.is_alive():
                        break
                    continue
                
                frame_lines, timestamp = frame_data
                
                # Display frame
                if self._on_frame:
                    self._on_frame(frame_lines)
                else:
                    self._display_frame(frame_lines)
                
                # Update metrics
                self.metrics.record_frame()
                self.metrics.update_fps()
                frame_count += 1
                
                # Frame timing
                if self._audio_sync:
                    self._audio_sync.frame_rendered()
                    sleep_time = self._audio_sync.get_sleep_time()
                else:
                    expected_time = start_time + frame_count * frame_duration
                    sleep_time = expected_time - time.time()
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            pass
        finally:
            self._stop_event.set()
            capture_thread.join(timeout=2)
    
    def _capture_loop(self, target_width: int, target_height: int) -> None:
        """Background capture and processing loop."""
        frame_count = 0
        
        while not self._stop_event.is_set():
            ret, frame = self._capture.read()
            
            if not ret:
                break
            
            timestamp = frame_count / self._source_fps
            
            # Process frame
            start = time.time()
            processed = self._processor.process_frame(
                frame,
                target_width,
                target_height,
            )
            
            # Render to characters
            art_lines = self._char_renderer.render(
                processed,
                show_edges=self.config.show_edges,
            )
            
            # Add frame wrapper
            frame_lines = self._frame_renderer.render_frame(
                art_lines,
                title=self._source_info.title if self._source_info else "",
                metrics=self.metrics,
                params=self._render_params,
            )
            
            render_time = time.time() - start
            self.metrics.record_render(render_time)
            
            # Adaptive quality adjustment
            if self.config.adaptive_quality and self._quality_thresholds:
                self._adjust_quality(render_time * 1000)
            
            # Add to buffer (non-blocking)
            if not self._frame_buffer.put(frame_lines, timestamp, block=False):
                self.metrics.record_dropped()
            
            frame_count += 1
    
    def _prebuffer(self, target_width: int, target_height: int) -> None:
        """Pre-buffer frames before playback starts."""
        print(f"📦 Buffering {self.config.prebuffer_frames} frames...", end="", flush=True)
        
        frame_count = 0
        start_time = time.time()
        
        while frame_count < self.config.prebuffer_frames:
            if time.time() - start_time > self.config.prebuffer_timeout:
                print(f"\r⚠️  Buffer timeout after {frame_count} frames        ")
                break
            
            ret, frame = self._capture.read()
            if not ret:
                break
            
            timestamp = frame_count / self._source_fps
            
            # Process frame
            processed = self._processor.process_frame(
                frame,
                target_width,
                target_height,
            )
            
            # Render to characters
            art_lines = self._char_renderer.render(
                processed,
                show_edges=self.config.show_edges,
            )
            
            # Add frame wrapper
            frame_lines = self._frame_renderer.render_frame(
                art_lines,
                title=self._source_info.title if self._source_info else "",
                metrics=self.metrics,
                params=self._render_params,
            )
            
            self._frame_buffer.put(frame_lines, timestamp, block=True)
            frame_count += 1
            
            # Progress indicator
            progress = frame_count * 100 // self.config.prebuffer_frames
            print(f"\r📦 Buffering... {progress}%  ", end="", flush=True)
        
        print(f"\r✓ Buffered {frame_count} frames                    ")
    
    def _adjust_quality(self, render_time_ms: float) -> None:
        """Adjust quality based on render performance.
        
        Args:
            render_time_ms: Last render time in milliseconds
        """
        if not self._quality_thresholds or not self._render_params:
            return
        
        if render_time_ms > self._quality_thresholds.reduce_ms:
            if self._render_params.decrease_quality():
                self._processor.update_quality(self._render_params.quality_level)
        elif render_time_ms < self._quality_thresholds.improve_ms:
            if self._render_params.increase_quality():
                self._processor.update_quality(self._render_params.quality_level)
    
    def _display_frame(self, frame_lines: List[str]) -> None:
        """Display frame to terminal.
        
        Args:
            frame_lines: Rendered frame lines
        """
        # Move cursor to home position
        sys.stdout.write("\033[H")
        
        # Write frame
        sys.stdout.write("\n".join(frame_lines))
        sys.stdout.write("\n")
        sys.stdout.flush()
    
    def _clear_screen(self) -> None:
        """Clear terminal screen."""
        if os.name == 'posix':
            os.system('clear')
        else:
            os.system('cls')
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        self._running = False
        self._stop_event.set()
        
        if self._capture:
            self._capture.release()
            self._capture = None
        
        if self._audio_player:
            self._audio_player.stop()
            self._audio_player.cleanup()
            self._audio_player = None
        
        if self._processor:
            self._processor.shutdown()
            self._processor = None
        
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None
        
        if self._frame_buffer:
            self._frame_buffer.clear()
            self._frame_buffer = None


# ═══════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════

def stream(
    source: Union[str, int, Path],
    fps: Optional[float] = None,
    color: bool = True,
    audio: bool = True,
    quality: str = "standard",
    gradient: str = "standard",
    show_stats: bool = True,
    adaptive: bool = False,
    prebuffer: int = 30,
) -> None:
    """Stream video to terminal with glyph art rendering.
    
    Convenience function for quick streaming setup.
    
    Args:
        source: Video source (URL, file, webcam index)
        fps: Target FPS (None = match source)
        color: Enable ANSI colors
        audio: Enable audio playback
        quality: Quality preset (minimal, low, standard, high, maximum)
        gradient: Character gradient preset
        show_stats: Show performance statistics
        adaptive: Enable adaptive quality adjustment
        prebuffer: Frames to buffer before playback
    """
    quality_map = {
        "minimal": QualityLevel.MINIMAL,
        "low": QualityLevel.LOW,
        "standard": QualityLevel.STANDARD,
        "high": QualityLevel.HIGH,
        "maximum": QualityLevel.MAXIMUM,
    }
    
    config = StreamConfig(
        target_fps=fps,
        color_enabled=color,
        audio_enabled=audio,
        quality_level=quality_map.get(quality, QualityLevel.STANDARD),
        gradient=gradient,
        show_stats=show_stats,
        adaptive_quality=adaptive,
        prebuffer_frames=prebuffer,
    )
    
    engine = StreamEngine(config)
    engine.run(source)


def stream_youtube(
    url: str,
    resolution: Optional[int] = None,
    **kwargs: Any,
) -> None:
    """Stream YouTube video to terminal.
    
    Args:
        url: YouTube URL
        resolution: Preferred resolution (360, 480, 720, 1080)
        **kwargs: Additional arguments passed to stream()
    """
    # Extract with preferred resolution
    extractor = YouTubeExtractor()
    result = extractor.extract(url, resolution=resolution, include_audio=True)
    
    if not result.has_video:
        raise StreamExtractionError("Failed to extract YouTube video")
    
    # Use extracted FPS if available
    if result.fps and 'fps' not in kwargs:
        kwargs['fps'] = result.fps
    
    stream(result.video_url, **kwargs)


def stream_webcam(
    device: int = 0,
    **kwargs: Any,
) -> None:
    """Stream webcam to terminal.
    
    Args:
        device: Webcam device index
        **kwargs: Additional arguments passed to stream()
    """
    kwargs.setdefault('audio', False)  # Webcam typically has no audio
    kwargs.setdefault('prebuffer', 5)  # Minimal buffering for live
    stream(device, **kwargs)
