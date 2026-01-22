from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetConfComputeProtectedMemoryUsage(device):
    c_memory = c_nvmlMemory_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetConfComputeProtectedMemoryUsage')
    ret = fn(device, byref(c_memory))
    _nvmlCheckReturn(ret)
    return c_memory