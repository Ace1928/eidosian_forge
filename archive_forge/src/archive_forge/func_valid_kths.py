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
@register_jitable
def valid_kths(a, kth):
    """
    Returns a sorted, unique array of kth values which serve
    as indexers for partitioning the input array, a.

    If the absolute value of any of the provided values
    is greater than a.shape[-1] an exception is raised since
    we are partitioning along the last axis (per Numpy default
    behaviour).

    Values less than 0 are transformed to equivalent positive
    index values.
    """
    kth_array = _asarray(kth).astype(np.int64)
    if kth_array.ndim != 1:
        raise ValueError('kth must be scalar or 1-D')
    if np.any(np.abs(kth_array) >= a.shape[-1]):
        raise ValueError('kth out of bounds')
    out = np.empty_like(kth_array)
    for index, val in np.ndenumerate(kth_array):
        if val < 0:
            out[index] = val + a.shape[-1]
        else:
            out[index] = val
    return np.unique(out)