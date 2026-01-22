from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
@convertStrBytes
def nvmlVgpuTypeGetName(vgpuTypeId):
    c_name = create_string_buffer(NVML_DEVICE_NAME_BUFFER_SIZE)
    c_buffer_size = c_uint(NVML_DEVICE_NAME_BUFFER_SIZE)
    fn = _nvmlGetFunctionPointer('nvmlVgpuTypeGetName')
    ret = fn(vgpuTypeId, c_name, byref(c_buffer_size))
    _nvmlCheckReturn(ret)
    return c_name.value