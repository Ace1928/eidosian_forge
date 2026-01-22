from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetVgpuUtilization(handle, timeStamp):
    c_vgpu_count = c_uint(0)
    c_time_stamp = c_ulonglong(timeStamp)
    c_sample_value_type = _nvmlValueType_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetVgpuUtilization')
    ret = fn(handle, c_time_stamp, byref(c_sample_value_type), byref(c_vgpu_count), None)
    if ret == NVML_SUCCESS:
        return []
    elif ret == NVML_ERROR_INSUFFICIENT_SIZE:
        sampleArray = c_vgpu_count.value * c_nvmlVgpuInstanceUtilizationSample_t
        c_samples = sampleArray()
        ret = fn(handle, c_time_stamp, byref(c_sample_value_type), byref(c_vgpu_count), c_samples)
        _nvmlCheckReturn(ret)
        return c_samples[0:c_vgpu_count.value]
    else:
        raise NVMLError(ret)