from __future__ import annotations
import contextlib
import enum
import typing
from typing_extensions import Protocol, runtime_checkable
from .constants import BOX_SYMBOLS, SHADE_SYMBOLS, Sizing
from .widget_decoration import WidgetDecoration, WidgetError
def ensure_bounds(new_trim_top: int) -> int:
    return max(0, min(canv_rows - maxrow, new_trim_top))