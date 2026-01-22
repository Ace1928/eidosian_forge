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
@overload(np.unwrap)
def numpy_unwrap(p, discont=None, axis=-1, period=6.283185307179586):
    if not isinstance(axis, (int, types.Integer)):
        msg = 'The argument "axis" must be an integer'
        raise TypingError(msg)
    if not type_can_asarray(p):
        msg = 'The argument "p" must be array-like'
        raise TypingError(msg)
    if not isinstance(discont, (types.Integer, types.Float)) and (not cgutils.is_nonelike(discont)):
        msg = 'The argument "discont" must be a scalar'
        raise TypingError(msg)
    if not isinstance(period, (float, types.Number)):
        msg = 'The argument "period" must be a scalar'
        raise TypingError(msg)
    slice1 = (slice(1, None, None),)
    if isinstance(period, types.Number):
        dtype = np.result_type(as_dtype(p.dtype), as_dtype(period))
    else:
        dtype = np.result_type(as_dtype(p.dtype), np.float64)
    integer_input = np.issubdtype(dtype, np.integer)

    def impl(p, discont=None, axis=-1, period=6.283185307179586):
        if axis != -1:
            msg = 'Value for argument "axis" is not supported'
            raise ValueError(msg)
        p_init = np.asarray(p).astype(dtype)
        init_shape = p_init.shape
        last_axis = init_shape[-1]
        p_new = p_init.reshape((p_init.size // last_axis, last_axis))
        if discont is None:
            discont = period / 2
        if integer_input:
            interval_high, rem = divmod(period, 2)
            boundary_ambiguous = rem == 0
        else:
            interval_high = period / 2
            boundary_ambiguous = True
        interval_low = -interval_high
        for i in range(p_init.size // last_axis):
            row = p_new[i]
            dd = np.diff(row)
            ddmod = np.mod(dd - interval_low, period) + interval_low
            if boundary_ambiguous:
                ddmod = np.where((ddmod == interval_low) & (dd > 0), interval_high, ddmod)
            ph_correct = ddmod - dd
            ph_correct = np.where(np.array([abs(x) for x in dd]) < discont, 0, ph_correct)
            ph_ravel = np.where(np.array([abs(x) for x in dd]) < discont, 0, ph_correct)
            ph_correct = np.reshape(ph_ravel, ph_correct.shape)
            up = np.copy(row)
            up[slice1] = row[slice1] + ph_correct.cumsum()
            p_new[i] = up
        return p_new.reshape(init_shape)
    return impl