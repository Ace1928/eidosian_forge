from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetBrand(handle):
    c_type = _nvmlBrandType_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetBrand')
    ret = fn(handle, byref(c_type))
    _nvmlCheckReturn(ret)
    return c_type.value