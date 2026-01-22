from __future__ import annotations
import typing
import warnings
from urwid.canvas import CanvasOverlay, CompositeCanvas
from urwid.split_repr import remove_defaults
from .constants import (
from .container import WidgetContainerListContentsMixin, WidgetContainerMixin
from .filler import calculate_top_bottom_filler
from .padding import calculate_left_right_padding
from .widget import Widget, WidgetError, WidgetWarning
def top_w_size(self, size: tuple[int, int], left: int, right: int, top: int, bottom: int) -> tuple[()] | tuple[int] | tuple[int, int]:
    """Return the size to pass to top_w."""
    if self.width_type == WHSettings.PACK:
        return ()
    maxcol, maxrow = size
    if self.width_type != WHSettings.PACK and self.height_type == WHSettings.PACK:
        return (maxcol - left - right,)
    return (maxcol - left - right, maxrow - top - bottom)