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
def ends_visible(self, size: tuple[int, int], focus: bool=False) -> list[Literal['top', 'bottom']]:
    """
        Return a list that may contain ``'top'`` and/or ``'bottom'``.

        i.e. this function will return one of: [], [``'top'``],
        [``'bottom'``] or [``'top'``, ``'bottom'``].

        convenience function for checking whether the top and bottom
        of the list are visible
        """
    maxcol, maxrow = size
    result = []
    middle, top, bottom = self.calculate_visible((maxcol, maxrow), focus=focus)
    if middle is None:
        return ['top', 'bottom']
    trim_top, above = top
    trim_bottom, below = bottom
    if trim_bottom == 0:
        row_offset, _w, pos, rows, _c = middle
        row_offset += rows
        for _w, pos, rows in below:
            row_offset += rows
        if row_offset < maxrow or self._body.get_next(pos) == (None, None):
            result.append('bottom')
    if trim_top == 0:
        row_offset, _w, pos, _rows, _c = middle
        for _w, pos, rows in above:
            row_offset -= rows
        if self._body.get_prev(pos) == (None, None):
            result.insert(0, 'top')
    return result