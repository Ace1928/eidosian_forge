from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetGpuInstanceId(device):
    c_gpuInstanceId = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetGpuInstanceId')
    ret = fn(device, byref(c_gpuInstanceId))
    _nvmlCheckReturn(ret)
    return c_gpuInstanceId.value