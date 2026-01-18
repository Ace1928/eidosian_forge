import pytest

from falling_sand.engine import ui
from falling_sand.engine.ui import OverlayConfig, UiOverlay


def test_overlay_config_validation() -> None:
    with pytest.raises(ValueError, match="font_scale must be positive"):
        OverlayConfig(font_scale=0.0)
    with pytest.raises(ValueError, match="text_color must be RGBA"):
        OverlayConfig(text_color=(1.0, 1.0, 1.0))


def test_ui_overlay_missing_dependency() -> None:
    if ui.OnscreenText is not None:
        pytest.skip("Panda3D available; overlay should instantiate.")
    with pytest.raises(RuntimeError, match="Panda3D is not available"):
        UiOverlay()
