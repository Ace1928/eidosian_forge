import functools
import warnings
import numpy as np
from numba import jit, typeof
from numba.core import cgutils, types, serialize, sigutils, errors
from numba.core.extending import (is_jitted, overload_attribute,
from numba.core.typing import npydecl
from numba.core.typing.templates import AbstractTemplate, signature
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.np.ufunc import _internal
from numba.parfors import array_analysis
from numba.np.ufunc import ufuncbuilder
from numba.np import numpy_support
from typing import Callable
from llvmlite import ir
def impl_nd_axis_int(ufunc, array, axis=0, dtype=None, initial=None):
    if axis is None:
        raise ValueError("'axis' must be specified")
    if axis < 0:
        axis += array.ndim
    if axis < 0 or axis >= array.ndim:
        raise ValueError('Invalid axis')
    shape = tuple_slice(array.shape, axis)
    if initial is None and identity is None:
        r = np.empty(shape, dtype=nb_dtype)
        for idx, _ in np.ndenumerate(r):
            result_idx = tuple_slice_append(idx, axis, 0)
            r[idx] = array[result_idx]
    elif initial is None and identity is not None:
        r = np.full(shape, fill_value=identity, dtype=nb_dtype)
    else:
        r = np.full(shape, fill_value=initial, dtype=nb_dtype)
    view = r.ravel()
    if initial is None and identity is None:
        for idx, val in np.ndenumerate(array):
            if idx[axis] == 0:
                continue
            else:
                flat_pos = compute_flat_idx(r.strides, r.itemsize, idx, axis)
                lhs, rhs = (view[flat_pos], val)
                view[flat_pos] = ufunc(lhs, rhs)
    else:
        for idx, val in np.ndenumerate(array):
            if initial is None and identity is None and (idx[axis] == 0):
                continue
            flat_pos = compute_flat_idx(r.strides, r.itemsize, idx, axis)
            lhs, rhs = (view[flat_pos], val)
            view[flat_pos] = ufunc(lhs, rhs)
    return r