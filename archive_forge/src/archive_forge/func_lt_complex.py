import collections
import ctypes
import re
import numpy as np
from numba.core import errors, types
from numba.core.typing.templates import signature
from numba.np import npdatetime_helpers
from numba.core.errors import TypingError
from numba.core.cgutils import is_nonelike   # noqa: F401
def lt_complex(a, b):
    if np.isnan(a.real):
        if np.isnan(b.real):
            if np.isnan(a.imag):
                return False
            elif np.isnan(b.imag):
                return True
            else:
                return a.imag < b.imag
        else:
            return False
    elif np.isnan(b.real):
        return True
    elif np.isnan(a.imag):
        if np.isnan(b.imag):
            return a.real < b.real
        else:
            return False
    elif np.isnan(b.imag):
        return True
    else:
        if a.real < b.real:
            return True
        elif a.real == b.real:
            return a.imag < b.imag
        return False