from __future__ import annotations
from typing import Generator, Iterable, Union
from prompt_toolkit.buffer import Buffer
from .containers import (
from .controls import BufferControl, SearchBufferControl, UIControl
def find_all_windows(self) -> Generator[Window, None, None]:
    """
        Find all the :class:`.UIControl` objects in this layout.
        """
    for item in self.walk():
        if isinstance(item, Window):
            yield item