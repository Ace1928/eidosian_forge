from __future__ import annotations
from typing import Generator, Iterable, Union
from prompt_toolkit.buffer import Buffer
from .containers import (
from .controls import BufferControl, SearchBufferControl, UIControl
def find_all_controls(self) -> Iterable[UIControl]:
    for container in self.find_all_windows():
        yield container.content