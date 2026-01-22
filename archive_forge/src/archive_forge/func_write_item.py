import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def write_item(item):
    k, v = item
    if k is None:
        self.write('**')
        self.set_precedence(_Precedence.EXPR, v)
        self.traverse(v)
    else:
        write_key_value_pair(k, v)