from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetSamples(device, sampling_type, timeStamp):
    c_sampling_type = _nvmlSamplingType_t(sampling_type)
    c_time_stamp = c_ulonglong(timeStamp)
    c_sample_count = c_uint(0)
    c_sample_value_type = _nvmlValueType_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetSamples')
    ret = fn(device, c_sampling_type, c_time_stamp, byref(c_sample_value_type), byref(c_sample_count), None)
    if ret != NVML_SUCCESS:
        raise NVMLError(ret)
    sampleArray = c_sample_count.value * c_nvmlSample_t
    c_samples = sampleArray()
    ret = fn(device, c_sampling_type, c_time_stamp, byref(c_sample_value_type), byref(c_sample_count), c_samples)
    _nvmlCheckReturn(ret)
    return (c_sample_value_type.value, c_samples[0:c_sample_count.value])