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
@overload(np.sinc)
def impl_np_sinc(x):
    if isinstance(x, types.Number):

        def impl(x):
            if x == 0.0:
                x = 1e-20
            x *= np.pi
            return np.sin(x) / x
        return impl
    elif isinstance(x, types.Array):

        def impl(x):
            out = np.zeros_like(x)
            for index, val in np.ndenumerate(x):
                out[index] = np.sinc(val)
            return out
        return impl
    else:
        raise NumbaTypeError('Argument "x" must be a Number or array-like.')