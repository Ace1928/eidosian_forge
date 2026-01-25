"""Audio playback and synchronization for streaming.

This module provides audio playback synchronized with video rendering,
supporting various audio sources and formats.

Classes:
    AudioPlayer: Background audio playback with timing control
    AudioSync: Synchronize video rendering with audio playback
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Callable

# Check for audio playback dependencies
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False

try:
    import simpleaudio
    HAS_SIMPLEAUDIO = True
except ImportError:
    HAS_SIMPLEAUDIO = False


# ═══════════════════════════════════════════════════════════════
# Audio Player
# ═══════════════════════════════════════════════════════════════

class AudioPlayer:
    """Background audio player with timing control.
    
    Provides non-blocking audio playback with position tracking
    for synchronization with video rendering.
    
    Attributes:
        source: Audio source path or URL
        position: Current playback position in seconds
        is_playing: Whether audio is currently playing
    """
    
    def __init__(self):
        """Initialize audio player."""
        self._source: Optional[str] = None
        self._temp_file: Optional[str] = None
        self._position: float = 0.0
        self._start_time: float = 0.0
        self._is_playing: bool = False
        self._paused: bool = False
        self._lock = threading.RLock()
        
        # Backend detection
        self._backend = self._detect_backend()
        
        # Playback control
        self._playback_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    @staticmethod
    def _detect_backend() -> str:
        """Detect available audio backend."""
        if HAS_PYGAME:
            return "pygame"
        elif HAS_SIMPLEAUDIO:
            return "simpleaudio"
        else:
            # Try ffplay as fallback
            try:
                subprocess.run(
                    ['ffplay', '-version'],
                    capture_output=True,
                    check=True,
                )
                return "ffplay"
            except Exception:
                pass
        return "none"
    
    @property
    def is_available(self) -> bool:
        """Check if audio playback is available."""
        return self._backend != "none"
    
    @property
    def position(self) -> float:
        """Current playback position in seconds."""
        with self._lock:
            if self._is_playing and not self._paused:
                return self._position + (time.time() - self._start_time)
            return self._position
    
    @property
    def is_playing(self) -> bool:
        """Whether audio is currently playing."""
        return self._is_playing and not self._paused
    
    def load(self, source: str, extract_audio: bool = False) -> bool:
        """Load audio source.
        
        Args:
            source: Audio file path or URL
            extract_audio: Whether to extract audio from video file
            
        Returns:
            True if loaded successfully
        """
        with self._lock:
            self.stop()
            
            if extract_audio and not source.endswith(('.mp3', '.wav', '.ogg', '.m4a')):
                # Extract audio to temp file
                self._temp_file = self._extract_audio(source)
                if self._temp_file:
                    self._source = self._temp_file
                    return True
                return False
            
            self._source = source
            return True
    
    def _extract_audio(self, video_path: str) -> Optional[str]:
        """Extract audio from video to temp file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to extracted audio, or None
        """
        try:
            fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            
            subprocess.run(
                [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-vn',
                    '-acodec', 'pcm_s16le',
                    '-ar', '44100',
                    '-ac', '2',
                    temp_path,
                ],
                capture_output=True,
                check=True,
                timeout=120,
            )
            
            return temp_path
        except Exception:
            return None
    
    def play(self, start_position: float = 0.0) -> bool:
        """Start audio playback.
        
        Args:
            start_position: Starting position in seconds
            
        Returns:
            True if playback started
        """
        if not self._source or self._backend == "none":
            return False
        
        with self._lock:
            self._position = start_position
            self._start_time = time.time()
            self._is_playing = True
            self._paused = False
            self._stop_event.clear()
        
        if self._backend == "pygame":
            return self._play_pygame(start_position)
        elif self._backend == "simpleaudio":
            return self._play_simpleaudio(start_position)
        elif self._backend == "ffplay":
            return self._play_ffplay(start_position)
        
        return False
    
    def _play_pygame(self, start_position: float) -> bool:
        """Play using pygame."""
        try:
            pygame.mixer.music.load(self._source)
            pygame.mixer.music.play(start=start_position)
            return True
        except Exception:
            return False
    
    def _play_simpleaudio(self, start_position: float) -> bool:
        """Play using simpleaudio (no seek support)."""
        # simpleaudio doesn't support seeking, start from beginning
        try:
            import wave
            wave_obj = simpleaudio.WaveObject.from_wave_file(self._source)
            wave_obj.play()
            return True
        except Exception:
            return False
    
    def _play_ffplay(self, start_position: float) -> bool:
        """Play using ffplay subprocess."""
        def _playback():
            try:
                cmd = [
                    'ffplay',
                    '-nodisp',
                    '-autoexit',
                    '-ss', str(start_position),
                    self._source,
                ]
                
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                
                while not self._stop_event.is_set():
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)
                
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                pass
        
        self._playback_thread = threading.Thread(target=_playback, daemon=True)
        self._playback_thread.start()
        return True
    
    def pause(self) -> None:
        """Pause playback."""
        with self._lock:
            if self._is_playing and not self._paused:
                self._position = self.position
                self._paused = True
                
                if self._backend == "pygame":
                    pygame.mixer.music.pause()
    
    def resume(self) -> None:
        """Resume playback."""
        with self._lock:
            if self._is_playing and self._paused:
                self._start_time = time.time()
                self._paused = False
                
                if self._backend == "pygame":
                    pygame.mixer.music.unpause()
    
    def stop(self) -> None:
        """Stop playback."""
        with self._lock:
            self._stop_event.set()
            self._is_playing = False
            self._paused = False
            self._position = 0.0
            
            if self._backend == "pygame":
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass
        
        # Wait for playback thread
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=1)
    
    def seek(self, position: float) -> None:
        """Seek to position.
        
        Args:
            position: Target position in seconds
        """
        was_playing = self.is_playing
        self.stop()
        
        if was_playing:
            self.play(start_position=position)
        else:
            with self._lock:
                self._position = position
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.stop()
        
        if self._temp_file and os.path.exists(self._temp_file):
            try:
                os.unlink(self._temp_file)
            except Exception:
                pass
            self._temp_file = None
    
    def __del__(self):
        """Destructor."""
        try:
            self.cleanup()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
# Audio Synchronization
# ═══════════════════════════════════════════════════════════════

class AudioSync:
    """Synchronize video rendering with audio playback.
    
    Provides timing control to keep video frames synchronized
    with audio playback, handling drift and buffering.
    
    Attributes:
        audio_player: Associated audio player
        sync_threshold: Maximum allowed drift in seconds
        target_fps: Target video frame rate
    """
    
    def __init__(
        self,
        audio_player: Optional[AudioPlayer] = None,
        sync_threshold: float = 0.1,
        target_fps: float = 30.0,
    ):
        """Initialize audio sync.
        
        Args:
            audio_player: Audio player to sync with
            sync_threshold: Max drift before correction
            target_fps: Target video frame rate
        """
        self.audio_player = audio_player
        self.sync_threshold = sync_threshold
        self.target_fps = target_fps
        self.frame_duration = 1.0 / target_fps
        
        self._start_time: float = 0.0
        self._frame_count: int = 0
        self._lock = threading.RLock()
    
    def start(self) -> None:
        """Start synchronization timing."""
        with self._lock:
            self._start_time = time.time()
            self._frame_count = 0
    
    def get_video_time(self) -> float:
        """Get current video time based on frame count.
        
        Returns:
            Current video time in seconds
        """
        with self._lock:
            return self._frame_count * self.frame_duration
    
    def get_audio_time(self) -> float:
        """Get current audio time.
        
        Returns:
            Current audio position in seconds
        """
        if self.audio_player and self.audio_player.is_playing:
            return self.audio_player.position
        return self.get_video_time()
    
    def get_drift(self) -> float:
        """Calculate drift between video and audio.
        
        Returns:
            Drift in seconds (positive = video ahead, negative = behind)
        """
        video_time = self.get_video_time()
        audio_time = self.get_audio_time()
        return video_time - audio_time
    
    def should_skip_frame(self) -> bool:
        """Check if frame should be skipped to catch up.
        
        Returns:
            True if frame should be skipped
        """
        drift = self.get_drift()
        return drift < -self.sync_threshold
    
    def get_sleep_time(self) -> float:
        """Get time to sleep before next frame.
        
        Accounts for audio sync and drift correction.
        
        Returns:
            Sleep time in seconds (0 if no sleep needed)
        """
        with self._lock:
            # Calculate ideal time for next frame
            ideal_time = self._start_time + (self._frame_count + 1) * self.frame_duration
            current_time = time.time()
            
            # Base sleep time
            sleep_time = ideal_time - current_time
            
            # Adjust for audio drift if syncing
            if self.audio_player and self.audio_player.is_playing:
                drift = self.get_drift()
                
                if drift > self.sync_threshold:
                    # Video is ahead, sleep more
                    sleep_time += drift * 0.5
                elif drift < -self.sync_threshold:
                    # Video is behind, sleep less
                    sleep_time -= abs(drift) * 0.5
            
            return max(0, sleep_time)
    
    def frame_rendered(self) -> None:
        """Signal that a frame has been rendered."""
        with self._lock:
            self._frame_count += 1
    
    def reset(self) -> None:
        """Reset synchronization."""
        with self._lock:
            self._start_time = time.time()
            self._frame_count = 0
    
    def wait_for_frame(self) -> bool:
        """Wait until it's time to render next frame.
        
        Returns:
            True if should render, False if should skip
        """
        if self.should_skip_frame():
            self.frame_rendered()  # Count as rendered (skipped)
            return False
        
        sleep_time = self.get_sleep_time()
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        return True


# ═══════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════

def check_audio_support() -> dict:
    """Check available audio support.
    
    Returns:
        Dictionary with support status for each backend
    """
    return {
        "pygame": HAS_PYGAME,
        "simpleaudio": HAS_SIMPLEAUDIO,
        "ffplay": AudioPlayer._detect_backend() == "ffplay" if not (HAS_PYGAME or HAS_SIMPLEAUDIO) else False,
        "any_available": HAS_PYGAME or HAS_SIMPLEAUDIO or AudioPlayer._detect_backend() != "none",
    }
