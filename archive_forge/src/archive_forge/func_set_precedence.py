import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def set_precedence(self, precedence, *nodes):
    for node in nodes:
        self._precedences[node] = precedence