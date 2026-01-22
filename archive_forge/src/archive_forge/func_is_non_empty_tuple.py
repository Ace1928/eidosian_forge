import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def is_non_empty_tuple(slice_value):
    return isinstance(slice_value, Tuple) and slice_value.elts