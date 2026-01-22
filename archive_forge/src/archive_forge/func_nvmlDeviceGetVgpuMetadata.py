from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetVgpuMetadata(handle):
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetVgpuMetadata')
    c_vgpuPgpuMetadata = c_nvmlVgpuPgpuMetadata_t()
    c_bufferSize = c_uint(0)
    ret = fn(handle, byref(c_vgpuPgpuMetadata), byref(c_bufferSize))
    if ret == NVML_ERROR_INSUFFICIENT_SIZE:
        ret = fn(handle, byref(c_vgpuPgpuMetadata), byref(c_bufferSize))
        _nvmlCheckReturn(ret)
    else:
        raise NVMLError(ret)
    return c_vgpuPgpuMetadata