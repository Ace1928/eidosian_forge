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
def get_cfarray_intrinsic(layout, dtype_):

    @intrinsic
    def intrinsic_cfarray(typingctx, ptr, shape):
        if ptr is types.voidptr:
            ptr_dtype = None
        elif isinstance(ptr, types.CPointer):
            ptr_dtype = ptr.dtype
        else:
            msg = f"pointer argument expected, got '{ptr}'"
            raise errors.NumbaTypeError(msg)
        if dtype_ is None:
            if ptr_dtype is None:
                msg = 'explicit dtype required for void* argument'
                raise errors.NumbaTypeError(msg)
            dtype = ptr_dtype
        elif isinstance(dtype_, types.DTypeSpec):
            dtype = dtype_.dtype
            if ptr_dtype is not None and dtype != ptr_dtype:
                msg = f"mismatching dtype '{dtype}' for pointer type '{ptr}'"
                raise errors.NumbaTypeError(msg)
        else:
            msg = f"invalid dtype spec '{dtype_}'"
            raise errors.NumbaTypeError(msg)
        ndim = ty_parse_shape(shape)
        if ndim is None:
            msg = f"invalid shape '{shape}'"
            raise errors.NumbaTypeError(msg)
        retty = types.Array(dtype, ndim, layout)
        sig = signature(retty, ptr, shape)
        return (sig, np_cfarray)
    return intrinsic_cfarray