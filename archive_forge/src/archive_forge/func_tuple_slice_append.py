import functools
import warnings
import numpy as np
from numba import jit, typeof
from numba.core import cgutils, types, serialize, sigutils, errors
from numba.core.extending import (is_jitted, overload_attribute,
from numba.core.typing import npydecl
from numba.core.typing.templates import AbstractTemplate, signature
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.np.ufunc import _internal
from numba.parfors import array_analysis
from numba.np.ufunc import ufuncbuilder
from numba.np import numpy_support
from typing import Callable
from llvmlite import ir
@register_jitable
def tuple_slice_append(tup, pos, val):
    s = tup_init
    i, j, sz = (0, 0, len(s))
    while j < sz:
        if j == pos:
            s = tuple_setitem(s, j, val)
        else:
            e = tup[i]
            s = tuple_setitem(s, j, e)
            i += 1
        j += 1
    return s