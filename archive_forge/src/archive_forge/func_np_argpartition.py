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
@overload(np.argpartition)
def np_argpartition(a, kth):
    if not isinstance(a, (types.Array, types.Sequence, types.Tuple)):
        raise TypeError('The first argument must be an array-like')
    if isinstance(a, types.Array) and a.ndim == 0:
        raise TypeError('The first argument must be at least 1-D (found 0-D)')
    kthdt = getattr(kth, 'dtype', kth)
    if not isinstance(kthdt, (types.Boolean, types.Integer)):
        raise TypeError('Partition index must be integer')

    def np_argpartition_impl(a, kth):
        a_tmp = _asarray(a)
        if a_tmp.size == 0:
            return a_tmp.copy().astype('intp')
        else:
            kth_array = valid_kths(a_tmp, kth)
            return np_argpartition_impl_inner(a_tmp, kth_array)
    return np_argpartition_impl