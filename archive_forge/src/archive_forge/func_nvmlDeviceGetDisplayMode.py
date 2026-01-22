from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetDisplayMode(handle):
    c_mode = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetDisplayMode')
    ret = fn(handle, byref(c_mode))
    _nvmlCheckReturn(ret)
    return c_mode.value