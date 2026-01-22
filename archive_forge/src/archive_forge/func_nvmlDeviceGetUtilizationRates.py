from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetUtilizationRates(handle):
    c_util = c_nvmlUtilization_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetUtilizationRates')
    ret = fn(handle, byref(c_util))
    _nvmlCheckReturn(ret)
    return c_util