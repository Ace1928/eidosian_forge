import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
def should_use_fpret(restype):
    """Determine if objc_msgSend_fpret is required to return a floating point type."""
    if not __i386__:
        return False
    if __LP64__ and restype == c_longdouble:
        return True
    if not __LP64__ and restype in (c_float, c_double, c_longdouble):
        return True
    return False