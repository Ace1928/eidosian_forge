import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial import polyutils as pu
from numba import literal_unroll
from numba.core import types, errors
from numba.core.extending import overload
from numba.np.numpy_support import type_can_asarray, as_dtype, from_dtype
@overload(poly.polyval, prefer_literal=True)
def poly_polyval(x, c, tensor=True):
    if not type_can_asarray(x):
        msg = 'The argument "x" must be array-like'
        raise errors.TypingError(msg)
    if not type_can_asarray(c):
        msg = 'The argument "c" must be array-like'
        raise errors.TypingError(msg)
    if not isinstance(tensor, (bool, types.BooleanLiteral)):
        msg = 'The argument "tensor" must be boolean'
        raise errors.RequireLiteralValue(msg)
    res_dtype = _poly_result_dtype(c, x)
    x_nd_array = not isinstance(x, types.Number)
    new_shape = (1,)
    if isinstance(x, types.Array):
        new_shape = (1,) * np.ndim(x)
    if isinstance(tensor, bool):
        tensor_arg = tensor
    else:
        tensor_arg = tensor.literal_value

    def impl(x, c, tensor=True):
        arr = np.asarray(c).astype(res_dtype)
        inputs = np.asarray(x).astype(res_dtype)
        if x_nd_array and tensor_arg:
            arr = arr.reshape(arr.shape + new_shape)
        l = len(arr)
        y = arr[l - 1] + inputs * 0
        for i in range(l - 1, 0, -1):
            y = arr[i - 1] + y * inputs
        return y
    return impl