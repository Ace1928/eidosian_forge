import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, errors, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.core.extending import overload_method, overload
from numba.misc import quicksort
from numba.cpython import slicing
from numba import literal_unroll
@overload_method(types.List, 'index')
def list_index(lst, value, start=0, stop=intp_max):
    if not isinstance(start, (int, types.Integer, types.Omitted)):
        raise errors.TypingError(f'arg "start" must be an Integer. Got {start}')
    if not isinstance(stop, (int, types.Integer, types.Omitted)):
        raise errors.TypingError(f'arg "stop" must be an Integer. Got {stop}')

    def list_index_impl(lst, value, start=0, stop=intp_max):
        n = len(lst)
        if start < 0:
            start += n
            if start < 0:
                start = 0
        if stop < 0:
            stop += n
        if stop > n:
            stop = n
        for i in range(start, stop):
            if lst[i] == value:
                return i
        raise ValueError('value not in list')
    return list_index_impl