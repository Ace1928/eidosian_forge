from array import array as native_array
import ctypes
import warnings
import operator
from functools import reduce # pylint: disable=redefined-builtin
import numpy as np
from ..base import _LIB, numeric_types, integer_types
from ..base import c_str, c_array, c_array_buf, c_handle_array, mx_real_t
from ..base import mx_uint, NDArrayHandle, check_call, DLPackHandle, mx_int, mx_int64
from ..base import ctypes2buffer
from ..runtime import Features
from ..context import Context, current_context
from ..util import is_np_array
from . import _internal
from . import op
from ._internal import NDArrayBase
def indexing_key_expand_implicit_axes(key, shape):
    """
    Make implicit axes explicit by adding ``slice(None)``
    and convert boolean array to integer array through `nonzero`.

    Examples
    --------
    >>> shape = (3, 4, 5)
    >>> indexing_key_expand_implicit_axes(np.s_[2, 1, 1], shape)
    (2, 1, 1)
    >>> indexing_key_expand_implicit_axes(np.s_[0], shape)
    (0, slice(None, None, None), slice(None, None, None))
    >>> indexing_key_expand_implicit_axes(np.s_[0, ...], shape)  # equivalent
    (0, slice(None, None, None), slice(None, None, None))
    >>> indexing_key_expand_implicit_axes(np.s_[:2, None, 0, ...], shape)
    (slice(None, 2, None), None, 0, slice(None, None, None))
    >>> bool_array = np.array([[True, False, True, False],
                               [False, True, False, True],
                               [True, False, True, False]], dtype=np.bool)
    >>> indexing_key_expand_implicit_axes(np.s_[bool_array, None, 0:2], shape)
    (array([0, 0, 1, 1, 2, 2], dtype=int64), array([0, 2, 1, 3, 0, 2], dtype=int64), None, slice(None, 2, None))
    """
    if not isinstance(key, tuple):
        key = (key,)
    ell_idx = None
    num_none = 0
    nonell_key = []
    prepend = _NDARRAY_NO_ZERO_DIM_BOOL_ARRAY
    axis = 0
    for i, idx in enumerate(key):
        if idx is Ellipsis:
            if ell_idx is not None:
                raise IndexError('Cannot use more than one ellipsis (`...`) for indexing')
            ell_idx = i
        else:
            if isinstance(idx, bool):
                idx = array(idx, dtype=np.bool_)
            if idx is None:
                num_none += 1
            if isinstance(idx, NDArrayBase) and idx.ndim == 0 and (idx.dtype == np.bool_):
                if not idx:
                    prepend = _NDARRAY_ZERO_DIM_BOOL_ARRAY_FALSE
                else:
                    prepend = _NDARRAY_ZERO_DIM_BOOL_ARRAY_TRUE
            elif isinstance(idx, NDArrayBase) and idx.ndim == 0 and (idx.dtype != np.bool_):
                nonell_key.append(int(idx.item()))
                axis += 1
            elif isinstance(idx, NDArrayBase) and idx.dtype == np.bool_:
                check_boolean_array_dimension(shape, axis, idx.shape)
                if not is_np_array():
                    raise ValueError('Cannot perform boolean indexing in legacy mode. Please activate numpy semantics by calling `npx.set_np()` in the global scope before calling this function.')
                nonell_key.extend(idx.nonzero())
                axis += idx.ndim
            else:
                nonell_key.append(idx)
                axis += 1
    nonell_key = tuple(nonell_key)
    if ell_idx is None:
        ell_idx = len(nonell_key)
    ell_ndim = len(shape) + num_none - len(nonell_key)
    expanded_key = nonell_key[:ell_idx] + (slice(None),) * ell_ndim + nonell_key[ell_idx:]
    return (expanded_key, prepend)