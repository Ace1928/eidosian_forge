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
@overload(np.where)
def ov_np_where_x_y(condition, x, y):
    if not type_can_asarray(condition):
        msg = 'The argument "condition" must be array-like'
        raise NumbaTypeError(msg)
    if is_nonelike(x) or is_nonelike(y):
        raise NumbaTypeError('Argument "x" or "y" cannot be None')
    for arg, name in zip((x, y), ('x', 'y')):
        if not type_can_asarray(arg):
            msg = 'The argument "{}" must be array-like if provided'
            raise NumbaTypeError(msg.format(name))
    cond_arr = isinstance(condition, types.Array)
    x_arr = isinstance(x, types.Array)
    y_arr = isinstance(y, types.Array)
    if cond_arr:
        x_dt = determine_dtype(x)
        y_dt = determine_dtype(y)
        dtype = np.promote_types(x_dt, y_dt)

        def check_0_dim(arg):
            return isinstance(arg, types.Number) or (isinstance(arg, types.Array) and arg.ndim == 0)
        special_0_case = all([check_0_dim(a) for a in (condition, x, y)])
        if special_0_case:
            return _where_zero_size_array_impl(dtype)
        layout = condition.layout
        if x_arr and y_arr:
            if x.layout == y.layout == condition.layout:
                layout = x.layout
            else:
                layout = 'A'
        return _where_generic_impl(dtype, layout)
    else:

        def impl(condition, x, y):
            return np.where(np.asarray(condition), np.asarray(x), np.asarray(y))
        return impl