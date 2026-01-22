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
def get_visible_amount(self, size: tuple[int, int], focus: bool=False) -> int:
    self._check_support_scrolling()
    if not self._body:
        return 1
    _mid, top, bottom = self.calculate_visible(size, focus)
    return 1 + len(top.fill) + len(bottom.fill)