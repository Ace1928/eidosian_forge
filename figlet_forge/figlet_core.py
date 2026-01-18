"""
Figlet Forge - Programmable text art and style transfer.
Provides Eidosian-style ASCII banners and visualizations.
"""
import logging
from typing import Dict, Any, List, Optional

class FigletForge:
    """
    Generates text-based art for headers and documentation.
    Supports multiple Eidosian-style ASCII frames.
    """
    def __init__(self):
        self.styles = {
            "simple": {"tl": "┌", "tr": "┐", "bl": "└", "br": "┘", "h": "─", "v": "│"},
            "bold": {"tl": "┏", "tr": "┓", "bl": "┗", "br": "┛", "h": "━", "v": "┃"},
            "double": {"tl": "╔", "tr": "╗", "bl": "╚", "br": "╝", "h": "═", "v": "║"},
            "elegant": {"tl": "╭", "tr": "╮", "bl": "╰", "br": "╯", "h": "─", "v": "│"},
        }

    def generate(self, text: str, style: str = "elegant") -> str:
        """Generate a styled text banner."""
        s = self.styles.get(style, self.styles["elegant"])
        width = len(text) + 4
        
        top = f"{s['tl']}{s['h'] * (width-2)}{s['tr']}"
        middle = f"{s['v']} {text} {s['v']}"
        bottom = f"{s['bl']}{s['h'] * (width-2)}{s['br']}"
        
        return f"{top}\n{middle}\n{bottom}"

    def render_header(self, text: str, style: str = "bold") -> str:
        """Render a high-visibility Eidosian header."""
        return self.generate(text.upper(), style=style)
