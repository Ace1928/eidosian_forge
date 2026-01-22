from __future__ import annotations
import typing
import warnings
from .constants import BAR_SYMBOLS, Align, Sizing, WrapMode
from .text import Text
from .widget import Widget
def set_completion(self, current: int) -> None:
    """
        current -- current progress
        """
    self._current = current
    self._invalidate()