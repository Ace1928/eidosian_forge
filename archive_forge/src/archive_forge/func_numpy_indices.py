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
@overload(np.indices)
def numpy_indices(dimensions):
    if not isinstance(dimensions, types.UniTuple):
        msg = 'The argument "dimensions" must be a tuple of integers'
        raise errors.TypingError(msg)
    if not isinstance(dimensions.dtype, types.Integer):
        msg = 'The argument "dimensions" must be a tuple of integers'
        raise errors.TypingError(msg)
    N = len(dimensions)
    shape = (1,) * N

    def impl(dimensions):
        res = np.empty((N,) + dimensions, dtype=np.int64)
        i = 0
        for dim in dimensions:
            idx = np.arange(dim, dtype=np.int64).reshape(tuple_setitem(shape, i, dim))
            res[i] = idx
            i += 1
        return res
    return impl