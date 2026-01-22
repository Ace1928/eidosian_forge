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
@overload(np.clip)
def np_clip(a, a_min, a_max, out=None):
    if not type_can_asarray(a):
        raise errors.TypingError('The argument "a" must be array-like')
    if not isinstance(a_min, types.NoneType) and (not type_can_asarray(a_min)):
        raise errors.TypingError('The argument "a_min" must be a number or an array-like')
    if not isinstance(a_max, types.NoneType) and (not type_can_asarray(a_max)):
        raise errors.TypingError('The argument "a_max" must be a number or an array-like')
    if not (isinstance(out, types.Array) or is_nonelike(out)):
        msg = 'The argument "out" must be an array if it is provided'
        raise errors.TypingError(msg)
    a_min_is_none = a_min is None or isinstance(a_min, types.NoneType)
    a_max_is_none = a_max is None or isinstance(a_max, types.NoneType)
    if a_min_is_none and a_max_is_none:

        def np_clip_nn(a, a_min, a_max, out=None):
            raise ValueError('array_clip: must set either max or min')
        return np_clip_nn
    a_min_is_scalar = isinstance(a_min, types.Number)
    a_max_is_scalar = isinstance(a_max, types.Number)
    if a_min_is_scalar and a_max_is_scalar:

        def np_clip_ss(a, a_min, a_max, out=None):
            ret = np.empty_like(a) if out is None else out
            for index in np.ndindex(a.shape):
                val_a = a[index]
                ret[index] = min(max(val_a, a_min), a_max)
            return ret
        return np_clip_ss
    elif a_min_is_scalar and (not a_max_is_scalar):
        if a_max_is_none:

            def np_clip_sn(a, a_min, a_max, out=None):
                ret = np.empty_like(a) if out is None else out
                for index in np.ndindex(a.shape):
                    val_a = a[index]
                    ret[index] = max(val_a, a_min)
                return ret
            return np_clip_sn
        else:

            def np_clip_sa(a, a_min, a_max, out=None):
                a_min_full = np.full_like(a, a_min)
                return _np_clip_impl(a, a_min_full, a_max, out)
            return np_clip_sa
    elif not a_min_is_scalar and a_max_is_scalar:
        if a_min_is_none:

            def np_clip_ns(a, a_min, a_max, out=None):
                ret = np.empty_like(a) if out is None else out
                for index in np.ndindex(a.shape):
                    val_a = a[index]
                    ret[index] = min(val_a, a_max)
                return ret
            return np_clip_ns
        else:

            def np_clip_as(a, a_min, a_max, out=None):
                a_max_full = np.full_like(a, a_max)
                return _np_clip_impl(a, a_min, a_max_full, out)
            return np_clip_as
    elif a_min_is_none:

        def np_clip_na(a, a_min, a_max, out=None):
            ret = np.empty_like(a) if out is None else out
            a_b, a_max_b = np.broadcast_arrays(a, a_max)
            return _np_clip_impl_none(a_b, a_max_b, True, ret)
        return np_clip_na
    elif a_max_is_none:

        def np_clip_an(a, a_min, a_max, out=None):
            ret = np.empty_like(a) if out is None else out
            a_b, a_min_b = np.broadcast_arrays(a, a_min)
            return _np_clip_impl_none(a_b, a_min_b, False, ret)
        return np_clip_an
    else:

        def np_clip_aa(a, a_min, a_max, out=None):
            return _np_clip_impl(a, a_min, a_max, out)
        return np_clip_aa