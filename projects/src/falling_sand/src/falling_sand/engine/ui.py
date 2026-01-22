"""Simple UI overlay utilities for Panda3D."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from direct.gui.OnscreenText import OnscreenText as OnscreenTextType  # type: ignore[import-not-found]
    from direct.gui.OnscreenText import OnscreenText  # type: ignore[import-not-found]
else:  # pragma: no cover - optional dependency
    try:
        from direct.gui.OnscreenText import OnscreenText
    except ImportError:
        OnscreenText = None


@dataclass
class OverlayConfig:
    """UI overlay configuration."""

    font_scale: float = 0.05
    text_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if self.font_scale <= 0:
            raise ValueError("font_scale must be positive")
        if len(self.text_color) != 4:
            raise ValueError("text_color must be RGBA")


class UiOverlay:
    """Overlay that renders a single text line."""

    def __init__(self, config: OverlayConfig | None = None) -> None:
        if OnscreenText is None:
            raise RuntimeError("Panda3D is not available. Install with 'pip install panda3d'.")
        self.config = config or OverlayConfig()
        self._text: "OnscreenTextType" = OnscreenText(
            text="",
            pos=(-1.3, 0.9),
            scale=self.config.font_scale,
            fg=self.config.text_color,
            align=0,
            mayChange=True,
        )

    def set_text(self, value: str) -> None:
        """Set overlay text."""

        self._text.setText(value)
