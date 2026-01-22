from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetRetiredPages_v2(device, sourceFilter):
    c_source = _nvmlPageRetirementCause_t(sourceFilter)
    c_count = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetRetiredPages_v2')
    ret = fn(device, c_source, byref(c_count), None)
    if ret != NVML_SUCCESS and ret != NVML_ERROR_INSUFFICIENT_SIZE:
        raise NVMLError(ret)
    c_count.value = c_count.value * 2 + 5
    page_array = c_ulonglong * c_count.value
    c_pages = page_array()
    times_array = c_ulonglong * c_count.value
    c_times = times_array()
    ret = fn(device, c_source, byref(c_count), c_pages, c_times)
    _nvmlCheckReturn(ret)
    return [{'address': int(c_pages[i]), 'timestamp': int(c_times[i])} for i in range(c_count.value)]