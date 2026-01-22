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
@overload(np.diagflat)
def numpy_diagflat(v, k=0):
    if not type_can_asarray(v):
        msg = 'The argument "v" must be array-like'
        raise errors.TypingError(msg)
    if not isinstance(k, (int, types.Integer)):
        msg = 'The argument "k" must be an integer'
        raise errors.TypingError(msg)

    def impl(v, k=0):
        v = np.asarray(v)
        v = v.ravel()
        s = len(v)
        abs_k = abs(k)
        n = s + abs_k
        res = np.zeros((n, n), v.dtype)
        i = np.maximum(0, -k)
        j = np.maximum(0, k)
        for t in range(s):
            res[i + t, j + t] = v[t]
        return res
    return impl