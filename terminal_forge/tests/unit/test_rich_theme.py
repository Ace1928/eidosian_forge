from __future__ import annotations

import pytest

from terminal_forge.rich_theme import build_eidosian_rich_theme, get_eidosian_palette


def test_palette_modes_are_distinct() -> None:
    dark = get_eidosian_palette("dark")
    light = get_eidosian_palette("light")
    assert dark.background != light.background
    assert dark.foreground != light.foreground


def test_build_theme_has_expected_style_keys() -> None:
    pytest.importorskip("rich")
    theme = build_eidosian_rich_theme("dark")
    for key in ("panel.border", "panel.title", "text.primary", "status.ok", "status.err"):
        assert key in theme.styles
