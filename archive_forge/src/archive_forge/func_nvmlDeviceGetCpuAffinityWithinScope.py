from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetCpuAffinityWithinScope(handle, cpuSetSize, scope):
    affinity_array = c_ulonglong * cpuSetSize
    c_affinity = affinity_array()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetCpuAffinityWithinScope')
    ret = fn(handle, cpuSetSize, byref(c_affinity), _nvmlAffinityScope_t(scope))
    _nvmlCheckReturn(ret)
    return c_affinity