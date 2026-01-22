from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetDefaultEccMode(handle):
    c_defaultState = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetDefaultEccMode')
    ret = fn(handle, byref(c_defaultState))
    _nvmlCheckReturn(ret)
    return [c_defaultState.value]