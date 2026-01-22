import collections
import ctypes
import re
import numpy as np
from numba.core import errors, types
from numba.core.typing.templates import signature
from numba.np import npdatetime_helpers
from numba.core.errors import TypingError
from numba.core.cgutils import is_nonelike   # noqa: F401
def farray(ptr, shape, dtype=None):
    """
    Return a Numpy array view over the data pointed to by *ptr* with the
    given *shape*, in Fortran order.  If *dtype* is given, it is used as the
    array's dtype, otherwise the array's dtype is inferred from *ptr*'s type.
    """
    if not isinstance(shape, int):
        shape = shape[::-1]
    return carray(ptr, shape, dtype).T