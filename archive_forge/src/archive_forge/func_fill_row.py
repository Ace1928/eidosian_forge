from __future__ import annotations
import typing
from urwid.canvas import CanvasCombine, CompositeCanvas, SolidCanvas
from urwid.util import get_encoding_mode
from .constants import BAR_SYMBOLS, Sizing
from .text import Text
from .widget import Widget, WidgetError, WidgetMeta, nocache_widget_render, nocache_widget_render_instance
def fill_row(row, chnum):
    rout = []
    for bar_type, width in row:
        if isinstance(bar_type, int) and len(self.hatt) > bar_type:
            rout.append(((bar_type, chnum), width))
            continue
        rout.append((bar_type, width))
    return rout