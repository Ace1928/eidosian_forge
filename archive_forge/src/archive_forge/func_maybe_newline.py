import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def maybe_newline(self):
    """Adds a newline if it isn't the start of generated source"""
    if self._source:
        self.write('\n')