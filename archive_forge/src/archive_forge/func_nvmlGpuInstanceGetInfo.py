from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlGpuInstanceGetInfo(gpuInstance):
    c_info = c_nvmlGpuInstanceInfo_t()
    fn = _nvmlGetFunctionPointer('nvmlGpuInstanceGetInfo')
    ret = fn(gpuInstance, byref(c_info))
    _nvmlCheckReturn(ret)
    return c_info