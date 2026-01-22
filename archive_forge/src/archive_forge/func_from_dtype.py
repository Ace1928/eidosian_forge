import collections
import ctypes
import re
import numpy as np
from numba.core import errors, types
from numba.core.typing.templates import signature
from numba.np import npdatetime_helpers
from numba.core.errors import TypingError
from numba.core.cgutils import is_nonelike   # noqa: F401
def from_dtype(dtype):
    """
    Return a Numba Type instance corresponding to the given Numpy *dtype*.
    NotImplementedError is raised on unsupported Numpy dtypes.
    """
    if type(dtype) is type and issubclass(dtype, np.generic):
        dtype = np.dtype(dtype)
    elif getattr(dtype, 'fields', None) is not None:
        return from_struct_dtype(dtype)
    try:
        return FROM_DTYPE[dtype]
    except KeyError:
        pass
    try:
        char = dtype.char
    except AttributeError:
        pass
    else:
        if char in 'SU':
            return _from_str_dtype(dtype)
        if char in 'mM':
            return _from_datetime_dtype(dtype)
        if char in 'V' and dtype.subdtype is not None:
            subtype = from_dtype(dtype.subdtype[0])
            return types.NestedArray(subtype, dtype.shape)
    raise errors.NumbaNotImplementedError(dtype)