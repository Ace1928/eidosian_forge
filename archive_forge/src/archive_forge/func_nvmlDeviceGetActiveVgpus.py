from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetActiveVgpus(handle):
    c_vgpu_count = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetActiveVgpus')
    ret = fn(handle, byref(c_vgpu_count), None)
    if ret == NVML_SUCCESS:
        return []
    elif ret == NVML_ERROR_INSUFFICIENT_SIZE:
        vgpu_instance_array = _nvmlVgpuInstance_t * c_vgpu_count.value
        c_vgpu_instances = vgpu_instance_array()
        ret = fn(handle, byref(c_vgpu_count), c_vgpu_instances)
        _nvmlCheckReturn(ret)
        vgpus = []
        for i in range(c_vgpu_count.value):
            vgpus.append(c_vgpu_instances[i])
        return vgpus
    else:
        raise NVMLError(ret)