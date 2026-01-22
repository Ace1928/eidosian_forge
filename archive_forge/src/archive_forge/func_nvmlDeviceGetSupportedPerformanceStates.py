from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetSupportedPerformanceStates(device):
    pstates = []
    c_count = c_uint(NVML_MAX_GPU_PERF_PSTATES)
    c_size = sizeof(c_uint) * c_count.value
    pstates_array = _nvmlPstates_t * c_count.value
    c_pstates = pstates_array()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetSupportedPerformanceStates')
    ret = fn(device, c_pstates, c_size)
    _nvmlCheckReturn(ret)
    for value in c_pstates:
        if value != NVML_PSTATE_UNKNOWN:
            pstates.append(value)
    return pstates