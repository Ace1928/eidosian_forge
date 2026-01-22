import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial import polyutils as pu
from numba import literal_unroll
from numba.core import types, errors
from numba.core.extending import overload
from numba.np.numpy_support import type_can_asarray, as_dtype, from_dtype
@overload(pu.as_series)
def polyutils_as_series(alist, trim=True):
    if not type_can_asarray(alist):
        msg = 'The argument "alist" must be array-like'
        raise errors.TypingError(msg)
    if not isinstance(trim, (bool, types.Boolean)):
        msg = 'The argument "trim" must be boolean'
        raise errors.TypingError(msg)
    res_dtype = np.float64
    tuple_input = isinstance(alist, types.BaseTuple)
    list_input = isinstance(alist, types.List)
    if tuple_input:
        if np.any(np.array([np.ndim(a) > 1 for a in alist])):
            raise errors.NumbaValueError('Coefficient array is not 1-d')
        res_dtype = _poly_result_dtype(*alist)
    elif list_input:
        dt = as_dtype(_get_list_type(alist))
        res_dtype = np.result_type(dt, np.float64)
    elif np.ndim(alist) <= 2:
        res_dtype = np.result_type(res_dtype, as_dtype(alist.dtype))
    else:
        raise errors.NumbaValueError('Coefficient array is not 1-d')

    def impl(alist, trim=True):
        if tuple_input:
            arrays = []
            for item in literal_unroll(alist):
                arrays.append(np.atleast_1d(np.asarray(item)).astype(res_dtype))
        elif list_input:
            arrays = [np.atleast_1d(np.asarray(a)).astype(res_dtype) for a in alist]
        else:
            alist_arr = np.asarray(alist)
            arrays = [np.atleast_1d(np.asarray(a)).astype(res_dtype) for a in alist_arr]
        if min([a.size for a in arrays]) == 0:
            raise ValueError('Coefficient array is empty')
        if trim:
            arrays = [pu.trimseq(a) for a in arrays]
        ret = arrays
        return ret
    return impl