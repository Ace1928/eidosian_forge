import collections
import ctypes
import re
import numpy as np
from numba.core import errors, types
from numba.core.typing.templates import signature
from numba.np import npdatetime_helpers
from numba.core.errors import TypingError
from numba.core.cgutils import is_nonelike   # noqa: F401
def map_arrayscalar_type(val):
    if isinstance(val, np.generic):
        dtype = val.dtype
    else:
        try:
            dtype = np.dtype(type(val))
        except TypeError:
            raise NotImplementedError('no corresponding numpy dtype for %r' % type(val))
    return from_dtype(dtype)