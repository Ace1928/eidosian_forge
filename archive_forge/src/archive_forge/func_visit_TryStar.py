import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def visit_TryStar(self, node):
    prev_in_try_star = self._in_try_star
    try:
        self._in_try_star = True
        self.do_visit_try(node)
    finally:
        self._in_try_star = prev_in_try_star