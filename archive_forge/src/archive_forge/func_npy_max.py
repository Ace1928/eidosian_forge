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
@overload(np.max)
@overload(np.amax)
@overload_method(types.Array, 'max')
def npy_max(a):
    if not isinstance(a, types.Array):
        return
    if isinstance(a.dtype, (types.NPDatetime, types.NPTimedelta)):
        pre_return_func = np.isnat
        comparator = max_comparator
    elif isinstance(a.dtype, types.Complex):
        pre_return_func = return_false

        def comp_func(a, max_val):
            if a.real > max_val.real:
                return True
            elif a.real == max_val.real:
                if a.imag > max_val.imag:
                    return True
            return False
        comparator = register_jitable(comp_func)
    elif isinstance(a.dtype, types.Float):
        pre_return_func = np.isnan
        comparator = max_comparator
    else:
        pre_return_func = return_false
        comparator = max_comparator

    def impl_max(a):
        if a.size == 0:
            raise ValueError('zero-size array to reduction operation maximum which has no identity')
        it = np.nditer(a)
        max_value = next(it).take(0)
        if pre_return_func(max_value):
            return max_value
        for view in it:
            v = view.item()
            if pre_return_func(v):
                return v
            if comparator(v, max_value):
                max_value = v
        return max_value
    return impl_max