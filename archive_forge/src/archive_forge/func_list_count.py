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
@overload_method(types.List, 'count')
def list_count(lst, value):

    def list_count_impl(lst, value):
        res = 0
        for elem in lst:
            if elem == value:
                res += 1
        return res
    return list_count_impl