import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
def c_array_buf(ctype, buf):
    """Create ctypes array from a Python buffer.
    For primitive types, using the buffer created with array.array is faster
    than a c_array call.

    Parameters
    ----------
    ctype : ctypes data type
        Data type of the array we want to convert to, such as mx_float.

    buf : buffer type
        Data content.

    Returns
    -------
    out : ctypes array
        Created ctypes array.

    Examples
    --------
    >>> x = mx.base.c_array_buf(mx.base.mx_float, array.array('i', [1, 2, 3]))
    >>> print len(x)
    3
    >>> x[1]
    2.0
    """
    return (ctype * len(buf)).from_buffer(buf)