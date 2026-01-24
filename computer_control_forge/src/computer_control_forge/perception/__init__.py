"""
Perception Module

Provides multi-modal screen perception:
- Visual: Screenshots, change detection, image analysis
- OCR: Text recognition with caching
- Windows: KWin-based window enumeration
- Integration: Unified perception state
"""

from .screen_state import (
    BoundingBox,
    PerceivedElement,
    ScreenFrame,
    ChangeRegion,
    ScreenCapture,
    ChangeDetector,
    ScreenPerception
)

from .ocr_engine import (
    TextRegion,
    OCRResult,
    OCREngine
)

from .window_manager import (
    WindowInfo,
    WorkspaceInfo,
    WindowManager
)

__all__ = [
    # Screen state
    'BoundingBox', 'PerceivedElement', 'ScreenFrame', 'ChangeRegion',
    'ScreenCapture', 'ChangeDetector', 'ScreenPerception',
    # OCR
    'TextRegion', 'OCRResult', 'OCREngine',
    # Windows
    'WindowInfo', 'WorkspaceInfo', 'WindowManager',
]

from .unified_perception import (
    PerceptionConfig,
    PerceptionState,
    SpatialIndex,
    UnifiedPerception,
    create_perception
)

__all__ += [
    'PerceptionConfig', 'PerceptionState', 'SpatialIndex',
    'UnifiedPerception', 'create_perception'
]
