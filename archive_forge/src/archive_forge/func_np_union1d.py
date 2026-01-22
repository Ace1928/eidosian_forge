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
@overload(np.union1d)
def np_union1d(ar1, ar2):
    if not type_can_asarray(ar1) or not type_can_asarray(ar2):
        raise TypingError('The arguments to np.union1d must be array-like')
    if ('unichr' in ar1.dtype.name or 'unichr' in ar2.dtype.name) and ar1.dtype.name != ar2.dtype.name:
        raise TypingError('For Unicode arrays, arrays must have same dtype')

    def union_impl(ar1, ar2):
        a = np.ravel(np.asarray(ar1))
        b = np.ravel(np.asarray(ar2))
        return np.unique(np.concatenate((a, b)))
    return union_impl