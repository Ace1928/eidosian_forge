from __future__ import annotations
import sys
from contextlib import contextmanager
from ctypes import windll
from ctypes.wintypes import HANDLE
from typing import Callable, ContextManager, Iterator
from prompt_toolkit.eventloop.win32 import create_win32_event
from ..key_binding import KeyPress
from ..utils import DummyContext
from .base import PipeInput
from .vt100_parser import Vt100Parser
from .win32 import _Win32InputBase, attach_win32_input, detach_win32_input
def typeahead_hash(self) -> str:
    """
        This needs to be unique for every `PipeInput`.
        """
    return f'pipe-input-{self._id}'