from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetMemoryInfo(handle, version=None):
    if not version:
        c_memory = c_nvmlMemory_t()
        fn = _nvmlGetFunctionPointer('nvmlDeviceGetMemoryInfo')
    else:
        c_memory = c_nvmlMemory_v2_t()
        c_memory.version = version
        fn = _nvmlGetFunctionPointer('nvmlDeviceGetMemoryInfo_v2')
    ret = fn(handle, byref(c_memory))
    _nvmlCheckReturn(ret)
    return c_memory