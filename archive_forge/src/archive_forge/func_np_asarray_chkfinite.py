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
@overload(np.asarray_chkfinite)
def np_asarray_chkfinite(a, dtype=None):
    msg = 'The argument to np.asarray_chkfinite must be array-like'
    if not isinstance(a, (types.Array, types.Sequence, types.Tuple)):
        raise TypingError(msg)
    if is_nonelike(dtype):
        dt = a.dtype
    else:
        try:
            dt = as_dtype(dtype)
        except NumbaNotImplementedError:
            raise TypingError('dtype must be a valid Numpy dtype')

    def impl(a, dtype=None):
        a = np.asarray(a, dtype=dt)
        for i in np.nditer(a):
            if not np.isfinite(i):
                raise ValueError('array must not contain infs or NaNs')
        return a
    return impl