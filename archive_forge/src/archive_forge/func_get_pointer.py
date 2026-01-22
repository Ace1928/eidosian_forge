from types import BuiltinFunctionType
import ctypes
from functools import partial
import numpy as np
from numba.core import types
from numba.core.errors import TypingError
from numba.core.typing import templates
from numba.np import numpy_support
def get_pointer(cffi_func):
    """
    Get a pointer to the underlying function for a CFFI function as an
    integer.
    """
    if cffi_func in _ool_func_ptr:
        return _ool_func_ptr[cffi_func]
    return int(ffi.cast('uintptr_t', cffi_func))