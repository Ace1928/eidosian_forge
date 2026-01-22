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
@overload(np.tril_indices_from)
def np_tril_indices_from(arr, k=0):
    check_is_integer(k, 'k')
    if arr.ndim != 2:
        raise TypingError('input array must be 2-d')

    def np_tril_indices_from_impl(arr, k=0):
        return np.tril_indices(arr.shape[0], k=k, m=arr.shape[1])
    return np_tril_indices_from_impl