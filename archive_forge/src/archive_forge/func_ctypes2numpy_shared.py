import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
def ctypes2numpy_shared(cptr, shape):
    """Convert a ctypes pointer to a numpy array.

    The resulting NumPy array shares the memory with the pointer.

    Parameters
    ----------
    cptr : ctypes.POINTER(mx_float)
        pointer to the memory region

    shape : tuple
        Shape of target `NDArray`.

    Returns
    -------
    out : numpy_array
        A numpy array : numpy array.
    """
    if not isinstance(cptr, ctypes.POINTER(mx_float)):
        raise RuntimeError('expected float pointer')
    size = 1
    for s in shape:
        size *= s
    dbuffer = (mx_float * size).from_address(ctypes.addressof(cptr.contents))
    return _np.frombuffer(dbuffer, dtype=_np.float32).reshape(shape)