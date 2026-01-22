from __future__ import annotations
import contextlib
import enum
import typing
from typing_extensions import Protocol, runtime_checkable
from .constants import BOX_SYMBOLS, SHADE_SYMBOLS, Sizing
from .widget_decoration import WidgetDecoration, WidgetError
@scrollbar_width.setter
def scrollbar_width(self, width: typing.SupportsInt) -> None:
    self._scrollbar_width = max(1, int(width))
    self._invalidate()