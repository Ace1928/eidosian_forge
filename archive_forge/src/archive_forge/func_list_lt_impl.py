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
def list_lt_impl(a, b):
    m = len(a)
    n = len(b)
    for i in range(min(m, n)):
        if a[i] < b[i]:
            return True
        elif a[i] > b[i]:
            return False
    return m < n