"""Glyph Stream: High-performance terminal rendering for video/camera streams.

This package provides modular streaming capabilities for the Glyph Forge system,
supporting real-time video, webcam, and YouTube streaming with Unicode art rendering.

Features:
    - High-performance frame processing with parallel pipelines
    - Pre-buffering for smooth playback at source frame rate
    - Audio synchronization (optional)
    - YouTube, webcam, local files, and network streams
    - ANSI color support (16/256/truecolor)
    - Multiple edge detection algorithms
    - Configurable character gradients

Modules:
    types: Core type definitions and enums
    extractors: Video source extraction (YouTube, files, webcams)
    processors: Frame processing and edge detection
    renderers: Character rendering and glyph mapping
    audio: Audio playback and synchronization
    engine: Main StreamEngine orchestrator
    ultimate: Best quality streaming with terminal recording

Example:
    >>> from glyph_forge.streaming import stream
    >>> stream("https://www.youtube.com/watch?v=...")
    
    >>> from glyph_forge.streaming import UltimateStreamEngine, UltimateConfig
    >>> config = UltimateConfig(render_mode='gradient', color_mode='truecolor')
    >>> engine = UltimateStreamEngine(config)
    >>> engine.run("video.mp4")
"""
from __future__ import annotations

from .types import (
    EdgeDetector,
    GradientResult,
    QualityLevel,
    VideoInfo,
    PerformanceStats,
    TextStyle,
    RenderThresholds,
    StreamMetrics,
    RenderParameters,
)
from .extractors import (
    YouTubeExtractor,
    VideoSourceExtractor,
    ExtractionResult,
    StreamExtractionError,
    DependencyError,
    AudioExtractor,
)
from .processors import (
    FrameProcessor,
    supersample_image,
    rgb_to_gray,
    detect_edges,
)
from .renderers import (
    CharacterRenderer,
    FrameRenderer,
    FrameBuffer,
    CHARACTER_MAPS,
)
from .audio import (
    AudioPlayer,
    AudioSync,
    check_audio_support,
)
from .engine import (
    StreamEngine,
    StreamConfig,
    stream,
    stream_youtube,
    stream_webcam,
)
from .hifi import (
    BrailleRenderer,
    ExtendedGradient,
    HybridRenderer,
    PerceptualColor,
    render_braille,
    render_hybrid,
    BRAILLE_CHARS,
    BLOCK_GRADIENT_64,
    EXTENDED_GRADIENT_128,
    STANDARD_RESOLUTIONS,
    resolution_to_terminal,
    terminal_to_resolution,
)
from .ultra import (
    UltraConfig,
    FramePool,
    DeltaEncoder,
    VectorizedANSI,
    UltraStreamEngine,
    stream_ultra,
    benchmark_rendering,
)
from .premium import (
    PremiumConfig,
    SmartBuffer,
    StreamRecorder,
    PremiumStreamEngine,
    stream_premium,
)
from .ultimate import (
    UltimateConfig,
    UltimateRenderer,
    UltimateStreamEngine,
    TerminalRecorder,
    LookupTables,
    get_lookup_tables,
    RenderCache,
    generate_output_name,
)

__all__ = [
    # Types
    "EdgeDetector",
    "GradientResult", 
    "QualityLevel",
    "VideoInfo",
    "PerformanceStats",
    "TextStyle",
    "RenderThresholds",
    "StreamMetrics",
    "RenderParameters",
    # Extractors
    "YouTubeExtractor",
    "VideoSourceExtractor",
    "ExtractionResult",
    "StreamExtractionError",
    "DependencyError",
    "AudioExtractor",
    # Processors
    "FrameProcessor",
    "supersample_image",
    "rgb_to_gray",
    "detect_edges",
    # Renderers
    "CharacterRenderer",
    "FrameRenderer",
    "FrameBuffer",
    "CHARACTER_MAPS",
    # Audio
    "AudioPlayer",
    "AudioSync",
    "check_audio_support",
    # Engine
    "StreamEngine",
    "StreamConfig",
    "stream",
    "stream_youtube",
    "stream_webcam",
    # High-Fidelity Rendering
    "BrailleRenderer",
    "ExtendedGradient",
    "HybridRenderer",
    "PerceptualColor",
    "render_braille",
    "render_hybrid",
    "BRAILLE_CHARS",
    "BLOCK_GRADIENT_64",
    "EXTENDED_GRADIENT_128",
    "STANDARD_RESOLUTIONS",
    "resolution_to_terminal",
    "terminal_to_resolution",
    # Ultra Performance
    "UltraConfig",
    "FramePool",
    "DeltaEncoder",
    "VectorizedANSI",
    "UltraStreamEngine",
    "stream_ultra",
    "benchmark_rendering",
    # Premium Streaming
    "PremiumConfig",
    "SmartBuffer",
    "StreamRecorder",
    "PremiumStreamEngine",
    "stream_premium",
    # Ultimate Streaming (Best Quality)
    "UltimateConfig",
    "UltimateRenderer",
    "UltimateStreamEngine",
    "TerminalRecorder",
    "LookupTables",
    "get_lookup_tables",
    "RenderCache",
    "generate_output_name",
]

__version__ = "1.0.0"
