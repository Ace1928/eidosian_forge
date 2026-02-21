"""Rich theme definitions for Terminal Forge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

try:
    from rich.theme import Theme
except Exception:  # pragma: no cover - optional runtime dependency
    Theme = None  # type: ignore[assignment]


@dataclass(frozen=True)
class EidosianPalette:
    background: str
    foreground: str
    accent: str
    success: str
    warning: str
    danger: str
    muted: str


def get_eidosian_palette(mode: str = "dark") -> EidosianPalette:
    """Return the canonical Eidosian terminal palette."""
    normalized = mode.lower().strip()
    if normalized == "light":
        return EidosianPalette(
            background="#f4f7fb",
            foreground="#1f2937",
            accent="#0f766e",
            success="#166534",
            warning="#a16207",
            danger="#b91c1c",
            muted="#6b7280",
        )
    return EidosianPalette(
        background="#0b1320",
        foreground="#e5e7eb",
        accent="#22d3ee",
        success="#34d399",
        warning="#f59e0b",
        danger="#f87171",
        muted="#94a3b8",
    )


def build_eidosian_rich_theme(mode: str = "dark") -> "Theme":
    """Build a Rich Theme object from the Eidosian palette."""
    if Theme is None:
        raise RuntimeError("rich is required to build terminal themes")
    palette = get_eidosian_palette(mode)
    styles: Dict[str, str] = {
        "panel.border": palette.accent,
        "panel.title": f"bold {palette.accent}",
        "text.primary": palette.foreground,
        "text.muted": palette.muted,
        "status.ok": f"bold {palette.success}",
        "status.warn": f"bold {palette.warning}",
        "status.err": f"bold {palette.danger}",
        "metric.key": palette.muted,
        "metric.value": f"bold {palette.foreground}",
        "highlight": f"bold {palette.accent}",
    }
    return Theme(styles)
