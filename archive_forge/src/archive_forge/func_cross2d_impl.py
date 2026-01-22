import math
from collections import namedtuple
import operator
import warnings
import llvmlite.ir
import numpy as np
from numba.core import types, cgutils
from numba.core.extending import overload, overload_method, register_jitable
from numba.np.numpy_support import (as_dtype, type_can_asarray, type_is_scalar,
from numba.core.imputils import (lower_builtin, impl_ret_borrowed,
from numba.np.arrayobj import (make_array, load_item, store_item,
from numba.np.linalg import ensure_blas
from numba.core.extending import intrinsic
from numba.core.errors import (RequireLiteralValue, TypingError,
from numba.cpython.unsafe.tuple import tuple_setitem
@overload(cross2d)
def cross2d_impl(a, b):
    if not type_can_asarray(a) or not type_can_asarray(b):
        raise TypingError('Inputs must be array-like.')

    def impl(a, b):
        a_ = np.asarray(a)
        b_ = np.asarray(b)
        if a_.shape[-1] != 2 or b_.shape[-1] != 2:
            raise ValueError('Incompatible dimensions for 2D cross product\n(dimension must be 2 for both inputs)')
        return _cross2d_operation(a_, b_)
    return impl