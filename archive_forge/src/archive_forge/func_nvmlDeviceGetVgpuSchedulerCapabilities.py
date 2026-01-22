from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetVgpuSchedulerCapabilities(handle):
    c_vgpu_sched_caps = c_nvmlVgpuSchedulerCapabilities_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetVgpuSchedulerCapabilities')
    ret = fn(handle, byref(c_vgpu_sched_caps))
    _nvmlCheckReturn(ret)
    return c_vgpu_sched_caps