import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
def c_str_array(strings):
    """Create ctypes const char ** from a list of Python strings.

    Parameters
    ----------
    strings : list of string
        Python strings.

    Returns
    -------
    (ctypes.c_char_p * len(strings))
        A const char ** pointer that can be passed to C API.
    """
    arr = (ctypes.c_char_p * len(strings))()
    arr[:] = [s.encode('utf-8') for s in strings]
    return arr