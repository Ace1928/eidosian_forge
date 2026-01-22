from __future__ import annotations
import operator
import typing
import warnings
from collections.abc import Iterable, Sized
from contextlib import suppress
from typing_extensions import Protocol, runtime_checkable
from urwid import signals
from urwid.canvas import CanvasCombine, SolidCanvas
from .constants import Sizing, VAlign, WHSettings, normalize_valign
from .container import WidgetContainerMixin
from .filler import calculate_top_bottom_filler
from .monitored_list import MonitoredFocusList, MonitoredList
from .widget import Widget, nocache_widget_render_instance
def update_pref_col_from_focus(self, size: tuple[int, int]) -> None:
    """Update self.pref_col from the focus widget."""
    maxcol, _maxrow = size
    widget, _old_pos = self._body.get_focus()
    if widget is None:
        return
    pref_col = None
    if hasattr(widget, 'get_pref_col'):
        pref_col = widget.get_pref_col((maxcol,))
    if pref_col is None and hasattr(widget, 'get_cursor_coords'):
        coords = widget.get_cursor_coords((maxcol,))
        if isinstance(coords, tuple):
            pref_col, _y = coords
    if pref_col is not None:
        self.pref_col = pref_col