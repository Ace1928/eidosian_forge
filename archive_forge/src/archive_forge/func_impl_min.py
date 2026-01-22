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
def impl_min(a):
    if a.size == 0:
        raise ValueError('zero-size array to reduction operation minimum which has no identity')
    it = np.nditer(a)
    min_value = next(it).take(0)
    if pre_return_func(min_value):
        return min_value
    for view in it:
        v = view.item()
        if pre_return_func(v):
            return v
        if comparator(v, min_value):
            min_value = v
    return min_value