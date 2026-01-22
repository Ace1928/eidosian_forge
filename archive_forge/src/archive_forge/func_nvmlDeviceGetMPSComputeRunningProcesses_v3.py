from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetMPSComputeRunningProcesses_v3(handle):
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMPSComputeRunningProcesses_v3')
    ret = fn(handle, byref(c_count), None)
    if ret == NVML_SUCCESS:
        return []
    elif ret == NVML_ERROR_INSUFFICIENT_SIZE:
        c_count.value = c_count.value * 2 + 5
        proc_array = c_nvmlProcessInfo_t * c_count.value
        c_procs = proc_array()
        ret = fn(handle, byref(c_count), c_procs)
        _nvmlCheckReturn(ret)
        procs = []
        for i in range(c_count.value):
            obj = nvmlStructToFriendlyObject(c_procs[i])
            if obj.usedGpuMemory == NVML_VALUE_NOT_AVAILABLE_ulonglong.value:
                obj.usedGpuMemory = None
            procs.append(obj)
        return procs
    else:
        raise NVMLError(ret)