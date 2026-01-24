"""Virtual display helpers using pyvirtualdisplay."""

from typing import Optional
from eidosian_core import eidosian

try:
    from pyvirtualdisplay import Display
except Exception:  # pragma: no cover - optional dependency
    Display = None


@eidosian()
def start_virtual_display(width: int = 1024, height: int = 768) -> Optional[object]:
    """Start an invisible virtual display.

    Args:
        width: Display width in pixels.
        height: Display height in pixels.

    Returns:
        Display object or ``None`` if pyvirtualdisplay is unavailable.
    """
    if Display is None:
        return None
    display = Display(visible=False, size=(width, height))
    display.start()
    return display


@eidosian()
def stop_virtual_display(display: object) -> None:
    """Stop a previously started virtual display."""
    if hasattr(display, "stop"):
        display.stop()
