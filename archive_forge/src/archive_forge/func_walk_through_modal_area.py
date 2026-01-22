from __future__ import annotations
from typing import Generator, Iterable, Union
from prompt_toolkit.buffer import Buffer
from .containers import (
from .controls import BufferControl, SearchBufferControl, UIControl
def walk_through_modal_area(self) -> Iterable[Container]:
    """
        Walk through all the containers which are in the current 'modal' part
        of the layout.
        """
    root: Container = self.current_window
    while not root.is_modal() and root in self._child_to_parent:
        root = self._child_to_parent[root]
    yield from walk(root)