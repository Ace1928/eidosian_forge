from __future__ import annotations

from eidosian_core import eidosian

from ..core import tool
from ..state import ROOT_DIR

try:
    import sys

    sys.path.append(str(ROOT_DIR / "figlet_forge"))
    from figlet_core import FigletForge
except ImportError:
    # Fallback if the path logic is different
    from figlet_forge.figlet_core import FigletForge

_forge = FigletForge()


@eidosian()
@tool(name="figlet_generate", description="Generate a styled ASCII text banner.")
def figlet_generate(text: str, style: str = "elegant") -> str:
    """Generate a styled text banner (elegant, bold, simple, double)."""
    return _forge.generate(text, style=style)


@eidosian()
@tool(name="figlet_header", description="Generate a high-visibility Eidosian header.")
def figlet_header(text: str, style: str = "bold") -> str:
    """Render a high-visibility Eidosian header in uppercase."""
    return _forge.render_header(text, style=style)
