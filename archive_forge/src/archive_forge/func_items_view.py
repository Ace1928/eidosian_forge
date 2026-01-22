import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def items_view(self, traverser, items):
    """Traverse and separate the given *items* with a comma and append it to
        the buffer. If *items* is a single item sequence, a trailing comma
        will be added."""
    if len(items) == 1:
        traverser(items[0])
        self.write(',')
    else:
        self.interleave(lambda: self.write(', '), traverser, items)