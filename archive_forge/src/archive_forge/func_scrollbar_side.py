from __future__ import annotations
import contextlib
import enum
import typing
from typing_extensions import Protocol, runtime_checkable
from .constants import BOX_SYMBOLS, SHADE_SYMBOLS, Sizing
from .widget_decoration import WidgetDecoration, WidgetError
@scrollbar_side.setter
def scrollbar_side(self, side: Literal['left', 'right']) -> None:
    if side not in {SCROLLBAR_LEFT, SCROLLBAR_RIGHT}:
        raise ValueError(f'scrollbar_side must be "left" or "right", not {side!r}')
    self._scrollbar_side = side
    self._invalidate()