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
@overload(np.count_nonzero)
def np_count_nonzero(a, axis=None):
    if not type_can_asarray(a):
        raise TypingError('The argument to np.count_nonzero must be array-like')
    if is_nonelike(axis):

        def impl(a, axis=None):
            arr2 = np.ravel(a)
            return np.sum(arr2 != 0)
        return impl
    else:

        def impl(a, axis=None):
            arr2 = a.astype(np.bool_)
            return np.sum(arr2, axis=axis)
        return impl