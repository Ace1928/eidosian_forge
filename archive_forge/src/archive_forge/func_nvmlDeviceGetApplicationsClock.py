from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetApplicationsClock(handle, type):
    c_clock = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetApplicationsClock')
    ret = fn(handle, _nvmlClockType_t(type), byref(c_clock))
    _nvmlCheckReturn(ret)
    return c_clock.value