"""Glyph Stream: High-performance terminal rendering.

This package provides the core streaming capabilities for Glyph Forge,
powered by the Unified Eidosian Stream Engine.
"""
from __future__ import annotations

from .engine import (
    UnifiedStreamEngine,
    UnifiedStreamConfig,
)
from .extractors import (
    VideoSourceExtractor,
    YouTubeExtractor,
    ExtractionResult,
    StreamExtractionError,
)

# Aliases for convenience
StreamEngine = UnifiedStreamEngine
StreamConfig = UnifiedStreamConfig

__all__ = [
    "UnifiedStreamEngine",
    "UnifiedStreamConfig",
    "StreamEngine",
    "StreamConfig",
    "VideoSourceExtractor",
    "YouTubeExtractor",
    "ExtractionResult",
    "StreamExtractionError",
]

__version__ = "2.0.0"