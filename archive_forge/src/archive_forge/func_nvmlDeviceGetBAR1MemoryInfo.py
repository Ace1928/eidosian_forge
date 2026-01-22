from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetBAR1MemoryInfo(handle):
    c_bar1_memory = c_nvmlBAR1Memory_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetBAR1MemoryInfo')
    ret = fn(handle, byref(c_bar1_memory))
    _nvmlCheckReturn(ret)
    return c_bar1_memory