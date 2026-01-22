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
def np_tril_impl_multi(m, k=0):
    mask = np.tri(m.shape[-2], M=m.shape[-1], k=k).astype(np.uint)
    idx = np.ndindex(m.shape[:-2])
    z = np.empty_like(m)
    zero_opt = np.zeros_like(mask, dtype=m.dtype)
    for sel in idx:
        z[sel] = np.where(mask, m[sel], zero_opt)
    return z