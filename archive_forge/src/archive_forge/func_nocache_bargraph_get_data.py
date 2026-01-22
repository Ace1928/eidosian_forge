from __future__ import annotations
import typing
from urwid.canvas import CanvasCombine, CompositeCanvas, SolidCanvas
from urwid.util import get_encoding_mode
from .constants import BAR_SYMBOLS, Sizing
from .text import Text
from .widget import Widget, WidgetError, WidgetMeta, nocache_widget_render, nocache_widget_render_instance
def nocache_bargraph_get_data(self, get_data_fn):
    """
    Disable caching on this bargraph because get_data_fn needs
    to be polled to get the latest data.
    """
    self.render = nocache_widget_render_instance(self)
    self._get_data = get_data_fn