from __future__ import annotations
from typing import Generator, Iterable, Union
from prompt_toolkit.buffer import Buffer
from .containers import (
from .controls import BufferControl, SearchBufferControl, UIControl
def get_visible_focusable_windows(self) -> list[Window]:
    """
        Return a list of :class:`.Window` objects that are focusable.
        """
    visible_windows = self.visible_windows
    return [w for w in self.get_focusable_windows() if w in visible_windows]