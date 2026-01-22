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
@overload(np.array_split)
def np_array_split(ary, indices_or_sections, axis=0):
    if isinstance(ary, (types.UniTuple, types.ListType, types.List)):

        def impl(ary, indices_or_sections, axis=0):
            return np.array_split(np.asarray(ary), indices_or_sections, axis=axis)
        return impl
    if isinstance(indices_or_sections, types.Integer):

        def impl(ary, indices_or_sections, axis=0):
            l, rem = divmod(ary.shape[axis], indices_or_sections)
            indices = np.cumsum(np.array([l + 1] * rem + [l] * (indices_or_sections - rem - 1)))
            return np.array_split(ary, indices, axis=axis)
        return impl
    elif isinstance(indices_or_sections, types.IterableType) and isinstance(indices_or_sections.iterator_type.yield_type, types.Integer):

        def impl(ary, indices_or_sections, axis=0):
            slice_tup = build_full_slice_tuple(ary.ndim)
            axis = normalize_axis('np.split', 'axis', ary.ndim, axis)
            out = []
            prev = 0
            for cur in indices_or_sections:
                idx = tuple_setitem(slice_tup, axis, slice(prev, cur))
                out.append(ary[idx])
                prev = cur
            out.append(ary[tuple_setitem(slice_tup, axis, slice(cur, None))])
            return out
        return impl
    elif isinstance(indices_or_sections, types.Tuple) and all((isinstance(t, types.Integer) for t in indices_or_sections.types)):

        def impl(ary, indices_or_sections, axis=0):
            slice_tup = build_full_slice_tuple(ary.ndim)
            axis = normalize_axis('np.split', 'axis', ary.ndim, axis)
            out = []
            prev = 0
            for cur in literal_unroll(indices_or_sections):
                idx = tuple_setitem(slice_tup, axis, slice(prev, cur))
                out.append(ary[idx])
                prev = cur
            out.append(ary[tuple_setitem(slice_tup, axis, slice(cur, None))])
            return out
        return impl