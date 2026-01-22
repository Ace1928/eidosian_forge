import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
def c_handle_array(objs):
    """Create ctypes const void ** from a list of MXNet objects with handles.

    Parameters
    ----------
    objs : list of NDArray/Symbol.
        MXNet objects.

    Returns
    -------
    (ctypes.c_void_p * len(objs))
        A void ** pointer that can be passed to C API.
    """
    arr = (ctypes.c_void_p * len(objs))()
    arr[:] = [o.handle for o in objs]
    return arr