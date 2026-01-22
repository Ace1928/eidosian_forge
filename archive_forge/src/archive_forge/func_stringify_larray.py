import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def stringify_larray(x, params):
    name = f'x{id(x)}'
    if x._data is not None:
        params.setdefault(name, x._data)
    return name