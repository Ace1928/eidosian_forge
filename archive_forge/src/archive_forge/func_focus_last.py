from __future__ import annotations
from typing import Generator, Iterable, Union
from prompt_toolkit.buffer import Buffer
from .containers import (
from .controls import BufferControl, SearchBufferControl, UIControl
def focus_last(self) -> None:
    """
        Give the focus to the last focused control.
        """
    if len(self._stack) > 1:
        self._stack = self._stack[:-1]