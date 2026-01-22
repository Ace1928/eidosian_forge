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
def get_first_visible_pos(self, size: tuple[int, int], focus: bool=False) -> int:
    self._check_support_scrolling()
    if not self._body:
        return 0
    _mid, top, _bottom = self.calculate_visible(size, focus)
    if top.fill:
        first_pos = top.fill[-1].position
    else:
        first_pos = self.focus_position
    over = 0
    _widget, first_pos = self.body.get_prev(first_pos)
    while first_pos is not None:
        over += 1
        _widget, first_pos = self.body.get_prev(first_pos)
    return over