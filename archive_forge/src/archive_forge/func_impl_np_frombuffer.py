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
@overload(np.frombuffer)
def impl_np_frombuffer(buffer, dtype=float):
    _check_const_str_dtype('frombuffer', dtype)
    if not isinstance(buffer, types.Buffer) or buffer.layout != 'C':
        msg = f'Argument "buffer" must be buffer-like. Got {buffer}'
        raise errors.TypingError(msg)
    if dtype is float or (isinstance(dtype, types.Function) and dtype.typing_key is float) or is_nonelike(dtype):
        nb_dtype = types.double
    else:
        nb_dtype = ty_parse_dtype(dtype)
    if nb_dtype is not None:
        retty = types.Array(dtype=nb_dtype, ndim=1, layout='C', readonly=not buffer.mutable)
    else:
        msg = f'Cannot parse input types to function np.frombuffer({buffer}, {dtype})'
        raise errors.TypingError(msg)

    def impl(buffer, dtype=float):
        return np_frombuffer(buffer, dtype, retty)
    return impl