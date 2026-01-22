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
@overload(np.swapaxes)
def numpy_swapaxes(a, axis1, axis2):
    if not isinstance(axis1, (int, types.Integer)):
        raise errors.TypingError('The second argument "axis1" must be an integer')
    if not isinstance(axis2, (int, types.Integer)):
        raise errors.TypingError('The third argument "axis2" must be an integer')
    if not isinstance(a, types.Array):
        raise errors.TypingError('The first argument "a" must be an array')
    ndim = a.ndim
    axes_list = tuple(range(ndim))

    def impl(a, axis1, axis2):
        axis1 = normalize_axis('np.swapaxes', 'axis1', ndim, axis1)
        axis2 = normalize_axis('np.swapaxes', 'axis2', ndim, axis2)
        if axis1 < 0:
            axis1 += ndim
        if axis2 < 0:
            axis2 += ndim
        axes_tuple = tuple_setitem(axes_list, axis1, axis2)
        axes_tuple = tuple_setitem(axes_tuple, axis2, axis1)
        return np.transpose(a, axes_tuple)
    return impl