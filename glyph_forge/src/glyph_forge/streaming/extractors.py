"""Video/Audio source extraction for streaming.

This module handles extraction of video and audio streams from various sources
including YouTube, local files, webcams, and network streams.

Classes:
    YouTubeExtractor: Extract streams from YouTube URLs
    VideoSourceExtractor: Unified source extraction interface
    StreamExtractionError: Extraction failure exception
    DependencyError: Missing dependency exception
"""
from __future__ import annotations

import os
import re
import time
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass

try:
    import yt_dlp
    HAS_YT_DLP = True
except ImportError:
    HAS_YT_DLP = False

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


# ═══════════════════════════════════════════════════════════════
# Exceptions
# ═══════════════════════════════════════════════════════════════

class StreamExtractionError(Exception):
    """Stream extraction error with rich diagnostic context.
    
    Provides categorized error information with original exception tracking
    for intelligent recovery strategies and detailed error reporting.
    """
    
    def __init__(
        self,
        message: str,
        original: Optional[Exception] = None,
        category: str = "general",
    ) -> None:
        super().__init__(message)
        self.original = original
        self.category = category
        self.timestamp = datetime.now()
    
    def get_diagnostic_info(self) -> Dict[str, Any]:
        """Get comprehensive diagnostic information."""
        return {
            "message": str(self),
            "category": self.category,
            "timestamp": self.timestamp.isoformat(),
            "original_type": type(self.original).__name__ if self.original else None,
            "original_message": str(self.original) if self.original else None,
        }


class DependencyError(Exception):
    """Missing dependency error with actionable installation guidance."""
    
    def __init__(
        self,
        package: str,
        install_cmd: str,
        required_for: str = "",
    ) -> None:
        feature_info = f" (required for {required_for})" if required_for else ""
        super().__init__(f"Missing dependency: {package}{feature_info}")
        self.package = package
        self.install_cmd = install_cmd
        self.required_for = required_for
    
    def get_installation_instructions(self) -> str:
        """Get user-friendly installation instructions."""
        return f"Install {self.package}: {self.install_cmd}"


# ═══════════════════════════════════════════════════════════════
# Extraction Result Types
# ═══════════════════════════════════════════════════════════════

@dataclass
class ExtractionResult:
    """Result of stream extraction with metadata."""
    video_url: Optional[str] = None
    audio_url: Optional[str] = None
    title: str = "Unknown"
    duration: Optional[float] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    format: str = "unknown"
    is_live: bool = False
    
    @property
    def has_video(self) -> bool:
        return self.video_url is not None
    
    @property
    def has_audio(self) -> bool:
        return self.audio_url is not None


# ═══════════════════════════════════════════════════════════════
# YouTube Extractor
# ═══════════════════════════════════════════════════════════════

class YouTubeExtractor:
    """High-performance YouTube stream extractor with caching.
    
    Extracts video and audio streams from YouTube URLs with intelligent
    caching, format selection, and error recovery.
    """
    
    _cache: Dict[str, Tuple[ExtractionResult, float]] = {}
    _cache_ttl: int = 3600  # 1 hour
    _cache_max_size: int = 100
    
    # YouTube URL patterns
    YOUTUBE_PATTERNS = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})',
        r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})',
    ]
    
    @classmethod
    def is_youtube_url(cls, url: str) -> bool:
        """Check if URL is a valid YouTube URL."""
        return any(re.match(pattern, url) for pattern in cls.YOUTUBE_PATTERNS)
    
    @classmethod
    def extract_video_id(cls, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        for pattern in cls.YOUTUBE_PATTERNS:
            match = re.match(pattern, url)
            if match:
                return match.group(1)
        return None
    
    @classmethod
    def extract(
        cls,
        url: str,
        resolution: Optional[int] = None,
        include_audio: bool = True,
    ) -> ExtractionResult:
        """Extract video and audio streams from YouTube URL.
        
        Args:
            url: YouTube URL or video ID
            resolution: Preferred vertical resolution (None=auto)
            include_audio: Whether to extract audio stream
            
        Returns:
            ExtractionResult: Extracted stream information
            
        Raises:
            DependencyError: If yt-dlp is not available
            StreamExtractionError: If extraction fails
        """
        if not HAS_YT_DLP:
            raise DependencyError(
                "yt-dlp",
                "pip install yt-dlp",
                "YouTube streaming",
            )
        
        # Check cache
        cache_key = f"{url}:{resolution}:{include_audio}"
        current_time = time.time()
        cls._prune_cache(current_time)
        
        if cache_key in cls._cache:
            result, timestamp = cls._cache[cache_key]
            if current_time - timestamp < cls._cache_ttl:
                return result
        
        # Determine optimal resolution
        actual_resolution = resolution or cls._determine_optimal_resolution()
        
        # Video format: prefer direct https MP4 for OpenCV compatibility
        video_format = (
            f'best[height<={actual_resolution}][protocol=https][ext=mp4]/'
            f'best[height<={actual_resolution}][protocol=https]/'
            f'best[protocol=https]'
        )
        
        # Audio format: prefer best audio quality
        audio_format = 'bestaudio[ext=m4a]/bestaudio'
        
        result = ExtractionResult()
        
        # Extract video stream
        video_opts = {
            'format': video_format,
            'quiet': True,
            'skip_download': True,
            'no_warnings': True,
            'socket_timeout': 15,
        }
        
        for retry in range(3):
            try:
                with yt_dlp.YoutubeDL(video_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    
                    if info:
                        result.video_url = info.get('url')
                        result.title = info.get('title', 'Unknown')
                        result.duration = info.get('duration')
                        result.fps = info.get('fps')
                        result.width = info.get('width')
                        result.height = info.get('height')
                        result.format = info.get('format', 'unknown')
                        result.is_live = info.get('is_live', False)
                        break
                        
            except Exception as e:
                if retry < 2:
                    time.sleep(1 * (2 ** retry))
                    # Fallback to lower quality
                    video_opts['format'] = (
                        'best[height<=360][protocol=https]/'
                        'worst[protocol=https]/worst'
                    )
                else:
                    raise StreamExtractionError(
                        f"Failed to extract video stream: {e}",
                        original=e,
                        category=cls._categorize_error(e),
                    )
        
        # Extract audio stream separately if requested
        if include_audio and not result.is_live:
            audio_opts = {
                'format': audio_format,
                'quiet': True,
                'skip_download': True,
                'no_warnings': True,
                'socket_timeout': 15,
            }
            
            try:
                with yt_dlp.YoutubeDL(audio_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    if info:
                        result.audio_url = info.get('url')
            except Exception:
                # Audio extraction failure is non-fatal
                pass
        
        # Cache successful result
        cls._cache[cache_key] = (result, current_time)
        
        return result
    
    @classmethod
    def _prune_cache(cls, current_time: float) -> None:
        """Remove expired and excess cache entries."""
        expired = [
            k for k, (_, ts) in cls._cache.items()
            if current_time - ts > cls._cache_ttl
        ]
        for key in expired:
            cls._cache.pop(key, None)
        
        if len(cls._cache) > cls._cache_max_size:
            sorted_items = sorted(
                cls._cache.items(),
                key=lambda item: item[1][1],
                reverse=True,
            )
            cls._cache = dict(sorted_items[:cls._cache_max_size])
    
    @staticmethod
    def _determine_optimal_resolution() -> int:
        """Determine optimal resolution based on terminal size."""
        try:
            term_size = os.get_terminal_size()
            if term_size.lines > 60:
                return 720
            elif term_size.lines > 40:
                return 480
            else:
                return 360
        except Exception:
            return 480
    
    @staticmethod
    def _categorize_error(error: Exception) -> str:
        """Categorize extraction errors for retry strategies."""
        error_str = str(error).lower()
        
        if "429" in error_str:
            return "rate_limited"
        elif "404" in error_str or "not found" in error_str:
            return "not_found"
        elif "private" in error_str:
            return "private"
        elif "age" in error_str or "restricted" in error_str:
            return "age_restricted"
        elif "timeout" in error_str or "timed out" in error_str:
            return "timeout"
        elif "network" in error_str or "connection" in error_str:
            return "network"
        else:
            return "general"


# ═══════════════════════════════════════════════════════════════
# Unified Video Source Extractor
# ═══════════════════════════════════════════════════════════════

class VideoSourceExtractor:
    """Unified interface for extracting video from any source.
    
    Handles YouTube URLs, local files, webcams, and network streams
    with automatic source type detection and appropriate extraction.
    """
    
    @classmethod
    def extract(
        cls,
        source: Union[str, int, Path],
        resolution: Optional[int] = None,
        include_audio: bool = True,
    ) -> ExtractionResult:
        """Extract video/audio from any supported source.
        
        Args:
            source: Video source (URL, file path, webcam index)
            resolution: Preferred resolution for streaming sources
            include_audio: Whether to extract audio
            
        Returns:
            ExtractionResult: Extracted stream information
        """
        # Handle webcam
        if isinstance(source, int):
            return cls._extract_webcam(source)
        
        source_str = str(source)
        
        # Handle YouTube
        if YouTubeExtractor.is_youtube_url(source_str):
            return YouTubeExtractor.extract(
                source_str,
                resolution=resolution,
                include_audio=include_audio,
            )
        
        # Handle local file
        if os.path.isfile(source_str):
            return cls._extract_local_file(source_str, include_audio)
        
        # Handle network stream (RTSP, HTTP, etc.)
        if source_str.startswith(('rtsp://', 'http://', 'https://')):
            return ExtractionResult(
                video_url=source_str,
                title=os.path.basename(source_str),
                format="network_stream",
            )
        
        raise StreamExtractionError(
            f"Unsupported source type: {source}",
            category="unsupported",
        )
    
    @classmethod
    def _extract_webcam(cls, index: int) -> ExtractionResult:
        """Extract webcam stream info."""
        if not HAS_OPENCV:
            raise DependencyError(
                "opencv-python",
                "pip install opencv-python",
                "webcam capture",
            )
        
        # Probe webcam for capabilities
        cap = cv2.VideoCapture(index)
        try:
            if not cap.isOpened():
                raise StreamExtractionError(
                    f"Cannot open webcam {index}",
                    category="device_error",
                )
            
            return ExtractionResult(
                video_url=str(index),
                title=f"Webcam {index}",
                fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None,
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None,
                format="webcam",
                is_live=True,
            )
        finally:
            cap.release()
    
    @classmethod
    def _extract_local_file(
        cls,
        path: str,
        include_audio: bool = True,
    ) -> ExtractionResult:
        """Extract local video file info."""
        if not HAS_OPENCV:
            raise DependencyError(
                "opencv-python",
                "pip install opencv-python",
                "video playback",
            )
        
        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                raise StreamExtractionError(
                    f"Cannot open video file: {path}",
                    category="file_error",
                )
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else None
            
            result = ExtractionResult(
                video_url=path,
                title=os.path.basename(path),
                duration=duration,
                fps=fps if fps > 0 else None,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None,
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None,
                format="local_file",
            )
            
            # Audio URL is same as video for local files (ffmpeg will extract)
            if include_audio:
                result.audio_url = path
            
            return result
        finally:
            cap.release()


# ═══════════════════════════════════════════════════════════════
# Audio Extractor
# ═══════════════════════════════════════════════════════════════

class AudioExtractor:
    """Extract and download audio for synchronized playback.
    
    Uses ffmpeg to extract audio streams to temporary files for
    synchronized playback with video rendering.
    """
    
    @staticmethod
    def has_ffmpeg() -> bool:
        """Check if ffmpeg is available."""
        try:
            subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                check=True,
            )
            return True
        except Exception:
            return False
    
    @classmethod
    def extract_to_file(
        cls,
        source: str,
        output_path: Optional[str] = None,
        start_time: float = 0,
    ) -> Optional[str]:
        """Extract audio to a temporary file.
        
        Args:
            source: Audio source URL or path
            output_path: Output path (None=auto temp file)
            start_time: Start time offset in seconds
            
        Returns:
            str: Path to extracted audio file, or None on failure
        """
        if not cls.has_ffmpeg():
            return None
        
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
        
        try:
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite
                '-ss', str(start_time),
                '-i', source,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',
                '-ar', '44100',
                '-ac', '2',
                output_path,
            ]
            
            subprocess.run(
                cmd,
                capture_output=True,
                check=True,
                timeout=60,
            )
            
            return output_path
        except Exception:
            if os.path.exists(output_path):
                os.unlink(output_path)
            return None
