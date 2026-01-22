import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
@functools.lru_cache(2 ** 14)
def find_full_reshape(newshape, size):
    try:
        expand = newshape.index(-1)
        before = newshape[:expand]
        after = newshape[expand + 1:]
        d = size // functools.reduce(operator.mul, itertools.chain(before, after), 1)
        return (*before, d, *after)
    except ValueError:
        return newshape