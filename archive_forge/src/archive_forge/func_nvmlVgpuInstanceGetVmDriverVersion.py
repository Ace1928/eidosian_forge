from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
@convertStrBytes
def nvmlVgpuInstanceGetVmDriverVersion(vgpuInstance):
    c_driver_version = create_string_buffer(NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE)
    c_buffer_size = c_uint(NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetVmDriverVersion')
    ret = fn(vgpuInstance, byref(c_driver_version), c_buffer_size)
    _nvmlCheckReturn(ret)
    return c_driver_version.value