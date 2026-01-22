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
def np_kaiser_impl(M, beta):
    if M < 1:
        return np.array((), dtype=np.float_)
    if M == 1:
        return np.ones(1, dtype=np.float_)
    n = np.arange(0, M)
    alpha = (M - 1) / 2.0
    return _i0n(n, alpha, beta)