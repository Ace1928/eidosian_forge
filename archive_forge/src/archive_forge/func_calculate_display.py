from __future__ import annotations
import typing
from urwid.canvas import CanvasCombine, CompositeCanvas, SolidCanvas
from urwid.util import get_encoding_mode
from .constants import BAR_SYMBOLS, Sizing
from .text import Text
from .widget import Widget, WidgetError, WidgetMeta, nocache_widget_render, nocache_widget_render_instance
def calculate_display(self, size: tuple[int, int]):
    """
        Calculate display data.
        """
    maxcol, maxrow = size
    bardata, top, hlines = self.get_data((maxcol, maxrow))
    widths = self.calculate_bar_widths((maxcol, maxrow), bardata)
    if self.use_smoothed():
        disp = calculate_bargraph_display(bardata, top, widths, maxrow * 8)
        disp = self.smooth_display(disp)
    else:
        disp = calculate_bargraph_display(bardata, top, widths, maxrow)
    if hlines:
        disp = self.hlines_display(disp, top, hlines, maxrow)
    return disp