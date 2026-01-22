import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def find_lazy(x):
    """Recursively search for ``LazyArray`` instances in pytrees."""
    if isinstance(x, LazyArray):
        yield x
        return
    if isinstance(x, (tuple, list)):
        for subx in x:
            yield from find_lazy(subx)
        return
    if isinstance(x, dict):
        for subx in x.values():
            yield from find_lazy(subx)
        return