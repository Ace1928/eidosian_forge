import __future__
from ast import PyCF_ONLY_AST
import codeop
import functools
import hashlib
import linecache
import operator
import time
from contextlib import contextmanager
@contextmanager
def extra_flags(self, flags):
    turn_on_bits = ~self.flags & flags
    self.flags = self.flags | flags
    try:
        yield
    finally:
        self.flags &= ~turn_on_bits