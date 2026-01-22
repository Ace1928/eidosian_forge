from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetMinorNumber(handle):
    c_minor_number = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMinorNumber')
    ret = fn(handle, byref(c_minor_number))
    _nvmlCheckReturn(ret)
    return c_minor_number.value