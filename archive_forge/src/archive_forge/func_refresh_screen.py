from abc import ABC, abstractmethod
from itertools import islice
from operator import itemgetter
from threading import RLock
from typing import (
from ._ratio import ratio_resolve
from .align import Align
from .console import Console, ConsoleOptions, RenderableType, RenderResult
from .highlighter import ReprHighlighter
from .panel import Panel
from .pretty import Pretty
from .region import Region
from .repr import Result, rich_repr
from .segment import Segment
from .style import StyleType
def refresh_screen(self, console: 'Console', layout_name: str) -> None:
    """Refresh a sub-layout.

        Args:
            console (Console): Console instance where Layout is to be rendered.
            layout_name (str): Name of layout.
        """
    with self._lock:
        layout = self[layout_name]
        region, _lines = self._render_map[layout]
        x, y, width, height = region
        lines = console.render_lines(layout, console.options.update_dimensions(width, height))
        self._render_map[layout] = LayoutRender(region, lines)
        console.update_screen_lines(lines, x, y)