from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetGpuMaxPcieLinkGeneration(handle):
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetGpuMaxPcieLinkGeneration')
    gen = c_uint()
    ret = fn(handle, byref(gen))
    _nvmlCheckReturn(ret)
    return gen.value