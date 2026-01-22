from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlGpuInstanceCreateComputeInstance(gpuInstance, profileId):
    c_instance = c_nvmlComputeInstance_t()
    fn = _nvmlGetFunctionPointer('nvmlGpuInstanceCreateComputeInstance')
    ret = fn(gpuInstance, profileId, byref(c_instance))
    _nvmlCheckReturn(ret)
    return c_instance