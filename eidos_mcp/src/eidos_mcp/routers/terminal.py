from __future__ import annotations

from typing import List, Optional, Tuple
from eidosian_core import eidosian
from ..core import tool
from ..state import FORGE_DIR

try:
    from terminal_forge import banner
except ImportError:
    import sys
    sys.path.append(str(FORGE_DIR / "terminal_forge/src"))
    from terminal_forge import banner

@eidosian()
@tool(name="terminal_banner", description="Generate a stylized terminal banner.")
def terminal_banner(
    text: str,
    title: str = "",
    style: str = "single",
    theme: str = "default",
    alignment: str = "left",
    width: int = 0
) -> str:
    """
    Create a modular terminal banner with borders and themes.
    
    Args:
        text: The main content of the banner.
        title: Optional banner title.
        style: 'single', 'double', 'rounded', 'bold', 'ascii'.
        theme: 'default', 'error', 'success', 'warning', 'info', 'cyberpunk', 'matrix'.
        alignment: 'left', 'center', 'right'.
        width: Optional fixed width.
    """
    b = banner.Banner(title=title, width=width)
    b.add_line(text)
    b.set_border(style)
    b.set_alignment(alignment)
    try:
        b = banner.Theme.apply(b, theme)
    except ValueError:
        pass
    return b.render()

@eidosian()
@tool(name="terminal_rainbow", description="Apply rainbow coloring to text.")
def terminal_rainbow(text: str) -> str:
    """Apply a rainbow ANSI color sequence to the provided text."""
    return banner.Color.rainbow(text)

@eidosian()
@tool(name="terminal_gradient", description="Apply smooth RGB gradient to text.")
def terminal_gradient(text: str, start_rgb: Tuple[int, int, int], end_rgb: Tuple[int, int, int]) -> str:
    """Apply a smooth RGB gradient transition to the text."""
    return banner.Color.gradient(text, start_rgb, end_rgb)
