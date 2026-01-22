import functools
import math
import operator
from llvmlite import ir
from llvmlite.ir import Constant
import numpy as np
from numba import pndindex, literal_unroll
from numba.core import types, typing, errors, cgutils, extending
from numba.np.numpy_support import (as_dtype, from_dtype, carray, farray,
from numba.np.numpy_support import type_can_asarray, is_nonelike, numpy_version
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core.typing import signature
from numba.core.types import StringLiteral
from numba.core.extending import (register_jitable, overload, overload_method,
from numba.misc import quicksort, mergesort
from numba.cpython import slicing
from numba.cpython.unsafe.tuple import tuple_setitem, build_full_slice_tuple
from numba.core.extending import overload_classmethod
from numba.core.typing.npydecl import (parse_dtype as ty_parse_dtype,
@extending.overload(np.eye)
def numpy_eye(N, M=None, k=0, dtype=float):
    if dtype is None or isinstance(dtype, types.NoneType):
        dt = np.dtype(float)
    elif isinstance(dtype, (types.DTypeSpec, types.Number)):
        dt = as_dtype(getattr(dtype, 'dtype', dtype))
    else:
        dt = np.dtype(dtype)

    def impl(N, M=None, k=0, dtype=float):
        _M = _eye_none_handler(N, M)
        arr = np.zeros((N, _M), dt)
        if k >= 0:
            d = min(N, _M - k)
            for i in range(d):
                arr[i, i + k] = 1
        else:
            d = min(N + k, _M)
            for i in range(d):
                arr[i - k, i] = 1
        return arr
    return impl