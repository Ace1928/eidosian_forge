from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetGpuInstanceById(device, gpuInstanceId):
    c_instance = c_nvmlGpuInstance_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetGpuInstanceById')
    ret = fn(device, gpuInstanceId, byref(c_instance))
    _nvmlCheckReturn(ret)
    return c_instance