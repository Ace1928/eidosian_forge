"""
Glyph Stream Engine - Main Orchestrator.

Brings together all components for seamless glyph streaming:
- VideoCapture for input
- AdaptiveBuffer for smooth playback
- GlyphRenderer for high-fidelity rendering
- GlyphRecorder for output recording
- AudioSync for synchronized audio

This is the main entry point for glyph streaming.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
import sys
import time
import signal
import cv2

from .config import StreamConfig, RenderMode, ColorMode, BufferStrategy
from .buffer import AdaptiveBuffer
from .capture import VideoCapture, VideoInfo
from .renderer import GlyphRenderer, RenderConfig
from .recorder import GlyphRecorder, RecorderConfig
from .sync import AudioSync, AudioConfig, AudioDownloader


@dataclass
class StreamStats:
    """Streaming statistics."""
    frames_displayed: int = 0
    frames_recorded: int = 0
    frames_dropped: int = 0
    avg_render_fps: float = 0.0
    buffer_fill: float = 0.0
    sync_offset: float = 0.0
    duration: float = 0.0


class GlyphStreamEngine:
    """High-performance glyph streaming engine.
    
    Orchestrates all components for smooth, high-quality glyph streaming
    with intelligent buffering, recording, and audio synchronization.
    
    Key features:
    - Adaptive buffering: Never starts playback until buffer is sufficient
    - Dual-resolution: Display scaled for terminal, recording at full HD
    - Smart sync: Drops frames for display if needed, records everything
    - Multiple sources: Files, YouTube, webcams, browser capture
    
    Usage:
        engine = GlyphStreamEngine()
        engine.stream('https://youtube.com/watch?v=...')
        
        # Or with custom config:
        config = StreamConfig(
            render_mode=RenderMode.GRADIENT,
            color_mode=ColorMode.ANSI256,
            record_output=True
        )
        engine = GlyphStreamEngine(config)
        engine.stream('video.mp4')
    """
    
    def __init__(self, config: Optional[StreamConfig] = None):
        """Initialize stream engine.
        
        Args:
            config: Stream configuration
        """
        self.config = config or StreamConfig()
        self.stats = StreamStats()
        
        # Components (initialized per stream)
        self._capture: Optional[VideoCapture] = None
        self._buffer: Optional[AdaptiveBuffer] = None
        self._renderer: Optional[GlyphRenderer] = None
        self._recorder: Optional[GlyphRecorder] = None
        self._audio: Optional[AudioSync] = None
        
        # State
        self._running = False
        self._start_time: Optional[float] = None
        
        # Signal handling
        self._setup_signals()
    
    def _setup_signals(self):
        """Setup signal handlers for clean shutdown."""
        def handler(signum, frame):
            print("\n⏹ Stopping...")
            self._running = False
        
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
    
    def stream(
        self,
        source: str,
        on_progress: Optional[Callable[[StreamStats], None]] = None
    ):
        """Stream video source with glyph rendering.
        
        Args:
            source: Video source (file, URL, webcam, etc.)
            on_progress: Optional callback for progress updates
        """
        self._running = True
        self.stats = StreamStats()
        
        try:
            # Phase 1: Setup
            print(f"🎬 Initializing stream: {source[:60]}...")
            self._setup_stream(source)
            
            # Phase 2: Buffer
            print(f"📦 Building buffer...")
            self._build_buffer()
            
            # Phase 3: Playback
            print(f"▶ Starting playback...")
            self._playback_loop(on_progress)
            
            # Phase 4: Finalize
            self._finalize()
            
        except KeyboardInterrupt:
            print("\n⏹ Interrupted")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            raise
        finally:
            self._cleanup()
    
    def _setup_stream(self, source: str):
        """Initialize all components for streaming."""
        # Check for cached output
        output_path = self.config.generate_output_path(source)
        
        if output_path.exists() and not self.config.force_rerender:
            print(f"✓ Found cached render: {output_path}")
            self._play_cached(output_path)
            return
        
        # Open video capture
        self._capture = VideoCapture(source, self.config.max_resolution)
        self._capture.open()
        
        info = self._capture.info
        print(f"  Source: {info.width}x{info.height} @ {info.fps:.1f}fps")
        print(f"  Duration: {info.duration:.1f}s ({info.total_frames} frames)")
        
        # Calculate display and recording sizes
        display_size = self.config.calculate_display_size(info.width, info.height)
        record_size = self.config.calculate_record_size(info.width, info.height)
        
        print(f"  Display: {display_size[0]}x{display_size[1]} chars")
        if self.config.record_output:
            print(f"  Recording: {record_size[0]}x{record_size[1]} chars (HD)")
        
        # Setup renderer
        render_config = RenderConfig(
            mode=self.config.render_mode.name.lower(),
            color=self.config.color_mode.name.lower(),
            dithering=self.config.dithering,
            cache_dir=self.config.cache_dir
        )
        self._renderer = GlyphRenderer(render_config)
        
        # Setup recorder
        if self.config.record_output:
            recorder_config = RecorderConfig(
                output_path=output_path,
                fps=info.fps
            )
            self._recorder = GlyphRecorder(recorder_config)
        
        # Setup audio
        if self.config.audio_enabled and info.audio_url:
            self._audio = AudioSync(AudioConfig(
                backend=self.config.audio_backend
            ))
        
        # Setup buffer
        target_fps = self.config.target_fps if self.config.target_fps > 0 else info.fps
        
        self._buffer = AdaptiveBuffer(
            target_fps=target_fps,
            min_buffer_seconds=self.config.min_buffer_seconds,
            target_buffer_seconds=self.config.target_buffer_seconds,
            max_buffer_frames=self.config.max_buffer_frames
        )
        
        # Store sizes for later
        self._display_size = display_size
        self._record_size = record_size if self.config.record_output else None
        self._video_info = info
        self._target_fps = target_fps
    
    def _build_buffer(self):
        """Start buffering and wait until ready for playback."""
        if not self._buffer or not self._capture or not self._renderer:
            return
        
        def frame_generator():
            return self._capture.read()
        
        def render_func(frame, w, h):
            return self._renderer.render(frame, w, h)
        
        def on_progress(buffered, total):
            pct = buffered / total * 100 if total > 0 else 0
            status = self._buffer.get_status()
            fps = status['render_fps']
            ready = "✓" if status['ready'] else "..."
            print(f"\r  Buffering: {pct:.0f}% ({buffered}/{total}) @ {fps:.0f}fps {ready}", 
                  end='', flush=True)
        
        self._buffer.start_buffering(
            frame_generator=frame_generator,
            render_func=render_func,
            display_size=self._display_size,
            record_size=self._record_size,
            total_frames=self._video_info.total_frames,
            on_progress=on_progress
        )
        
        # Wait for buffer to be ready
        while not self._buffer.ready_for_playback and self._running:
            time.sleep(0.1)
        
        print()  # New line after progress
        
        status = self._buffer.get_status()
        print(f"  Buffer ready: {status['buffer_seconds']:.1f}s ({status['buffer_size']} frames)")
    
    def _playback_loop(
        self,
        on_progress: Optional[Callable[[StreamStats], None]] = None
    ):
        """Main playback loop."""
        if not self._buffer:
            return
        
        # Clear screen
        if self.config.clear_screen:
            sys.stdout.write('\033[2J\033[H')
            sys.stdout.flush()
        
        # Start audio
        if self._audio and self._video_info.audio_url:
            print("🔊 Starting audio...")
            if self._video_info.audio_url.startswith(('http://', 'https://')):
                self._audio.start_stream(self._video_info.audio_url)
            else:
                self._audio.start(self._video_info.audio_url)
        
        self._start_time = time.perf_counter()
        frame_idx = 0
        
        while self._running and self._buffer.has_frames:
            loop_start = time.perf_counter()
            
            # Get next frame
            frame = self._buffer.get_next_frame(timeout=0.5)
            if frame is None:
                if self._buffer._buffering_complete:
                    break
                continue
            
            # Check sync - are we behind?
            elapsed = time.perf_counter() - self._start_time
            target_time = frame.timestamp
            drift = elapsed - target_time  # Positive = we're behind
            
            # If we're more than 0.5s behind, skip display frames (but still record)
            if drift > 0.5:
                # Record the frame
                if self._recorder:
                    self._recorder.write_frame(frame.record_data or frame.display_data)
                
                self.stats.frames_dropped += 1
                frame_idx += 1
                continue
            
            # Display
            sys.stdout.write('\033[H')  # Cursor home
            sys.stdout.write(frame.display_data)
            
            # Show metrics
            if self.config.show_metrics:
                status = self._buffer.get_status()
                sync = self._audio.get_sync_offset(frame.timestamp) if self._audio else 0
                
                metrics = (
                    f"\n\033[K[{frame_idx}/{self._video_info.total_frames}] "
                    f"buf:{status['buffer_size']} | "
                    f"sync:{sync:+.2f}s"
                )
                if self.stats.frames_dropped > 0:
                    metrics += f" | drop:{self.stats.frames_dropped}"
                
                sys.stdout.write(metrics)
            
            sys.stdout.flush()
            
            # Record
            if self._recorder:
                self._recorder.write_frame(frame.record_data or frame.display_data)
                self.stats.frames_recorded += 1
            
            self.stats.frames_displayed += 1
            frame_idx += 1
            
            # Timing - wait for next frame time
            elapsed = time.perf_counter() - self._start_time
            next_time = (frame_idx) / self._target_fps
            sleep_time = next_time - elapsed
            
            if sleep_time > 0.001:
                time.sleep(sleep_time)
            
            # Progress callback
            if on_progress:
                self.stats.buffer_fill = status['buffer_seconds']
                self.stats.sync_offset = sync if self._audio else 0
                self.stats.duration = elapsed
                on_progress(self.stats)
        
        self.stats.duration = time.perf_counter() - self._start_time
    
    def _finalize(self):
        """Finalize recording and mux audio."""
        print("\n\033[0m")  # Reset colors
        
        if self._recorder:
            self._recorder.close()
            
            # Mux audio if available
            if self.config.mux_audio and self._video_info.audio_url:
                print("🎵 Muxing audio...")
                
                # For URLs, we need to download audio first
                audio_path = self._video_info.audio_url
                
                if audio_path.startswith(('http://', 'https://')):
                    # Download audio
                    temp_audio = Path('/tmp/glyph_audio.m4a')
                    print("  Downloading audio...")
                    downloaded = AudioDownloader.download_youtube_audio(
                        self._capture.source if 'youtube' in self._capture.source else audio_path,
                        temp_audio
                    )
                    if downloaded:
                        audio_path = str(downloaded)
                    else:
                        print("  ⚠ Audio download failed")
                        return
                
                self._recorder.mux_audio(audio_path)
        
        # Print stats
        print(f"\n✓ Complete!")
        print(f"  Displayed: {self.stats.frames_displayed} frames")
        print(f"  Recorded: {self.stats.frames_recorded} frames")
        if self.stats.frames_dropped > 0:
            print(f"  Dropped: {self.stats.frames_dropped} frames (sync)")
        print(f"  Duration: {self.stats.duration:.1f}s")
    
    def _cleanup(self):
        """Cleanup all resources."""
        if self._buffer:
            self._buffer.stop()
        
        if self._audio:
            self._audio.stop()
        
        if self._capture:
            self._capture.close()
        
        if self._recorder and self._recorder.is_open:
            self._recorder.close()
    
    def _play_cached(self, cache_path: Path):
        """Play from cached render."""
        print(f"▶ Playing cached: {cache_path}")
        
        # Use system video player or ffplay
        import subprocess
        subprocess.run(['ffplay', '-autoexit', str(cache_path)], 
                      capture_output=True)
    
    def stop(self):
        """Stop streaming."""
        self._running = False


# Convenience function
def stream(source: str, **kwargs):
    """Quick stream function.
    
    Args:
        source: Video source
        **kwargs: StreamConfig parameters
    """
    config = StreamConfig(**kwargs)
    engine = GlyphStreamEngine(config)
    engine.stream(source)
