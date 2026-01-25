"""
Glyph Forge Streaming Core - Modular High-Performance Architecture

This package provides the core streaming infrastructure:
- config.py: Configuration management
- buffer.py: Smart adaptive buffering system
- capture.py: Video capture from URLs, files, webcams, browsers
- renderer.py: High-fidelity glyph rendering engine
- recorder.py: Terminal-to-video recording
- sync.py: Audio playback and synchronization
- engine.py: Main orchestrator

Usage:
    from glyph_forge.streaming.core import GlyphStreamEngine, StreamConfig
    
    engine = GlyphStreamEngine()
    engine.stream('https://youtube.com/watch?v=...')
"""

from .config import StreamConfig, RenderMode, ColorMode, BufferStrategy
from .buffer import AdaptiveBuffer, BufferedFrame, BufferMetrics
from .capture import VideoCapture, VideoInfo, CaptureSource
from .renderer import GlyphRenderer, RenderConfig, LookupTables
from .recorder import GlyphRecorder, RecorderConfig
from .sync import AudioSync, AudioConfig, AudioDownloader
from .engine import GlyphStreamEngine, StreamStats, stream

__all__ = [
    # Config
    'StreamConfig',
    'RenderMode',
    'ColorMode',
    'BufferStrategy',
    # Buffer
    'AdaptiveBuffer',
    'BufferedFrame',
    'BufferMetrics',
    # Capture
    'VideoCapture',
    'VideoInfo',
    'CaptureSource',
    # Renderer
    'GlyphRenderer',
    'RenderConfig',
    'LookupTables',
    # Recorder
    'GlyphRecorder',
    'RecorderConfig',
    # Audio
    'AudioSync',
    'AudioConfig',
    'AudioDownloader',
    # Engine
    'GlyphStreamEngine',
    'StreamStats',
    'stream',
]
