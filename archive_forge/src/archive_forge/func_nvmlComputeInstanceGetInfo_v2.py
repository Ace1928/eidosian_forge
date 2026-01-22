from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlComputeInstanceGetInfo_v2(computeInstance):
    c_info = c_nvmlComputeInstanceInfo_t()
    fn = _nvmlGetFunctionPointer('nvmlComputeInstanceGetInfo_v2')
    ret = fn(computeInstance, byref(c_info))
    _nvmlCheckReturn(ret)
    return c_info