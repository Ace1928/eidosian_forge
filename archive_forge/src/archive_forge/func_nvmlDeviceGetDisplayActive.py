from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetDisplayActive(handle):
    c_mode = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetDisplayActive')
    ret = fn(handle, byref(c_mode))
    _nvmlCheckReturn(ret)
    return c_mode.value