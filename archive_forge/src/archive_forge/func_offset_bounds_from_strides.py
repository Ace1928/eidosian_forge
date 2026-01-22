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
def offset_bounds_from_strides(context, builder, arrty, arr, shapes, strides):
    """
    Compute a half-open range [lower, upper) of byte offsets from the
    array's data pointer, that bound the in-memory extent of the array.

    This mimics offset_bounds_from_strides() from
    numpy/core/src/private/mem_overlap.c
    """
    itemsize = arr.itemsize
    zero = itemsize.type(0)
    one = zero.type(1)
    if arrty.layout in 'CF':
        lower = zero
        upper = builder.mul(itemsize, arr.nitems)
    else:
        lower = zero
        upper = zero
        for i in range(arrty.ndim):
            max_axis_offset = builder.mul(strides[i], builder.sub(shapes[i], one))
            is_upwards = builder.icmp_signed('>=', max_axis_offset, zero)
            upper = builder.select(is_upwards, builder.add(upper, max_axis_offset), upper)
            lower = builder.select(is_upwards, lower, builder.add(lower, max_axis_offset))
        upper = builder.add(upper, itemsize)
        is_empty = builder.icmp_signed('==', arr.nitems, zero)
        upper = builder.select(is_empty, zero, upper)
        lower = builder.select(is_empty, zero, lower)
    return (lower, upper)