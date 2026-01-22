from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
@convertStrBytes
def nvmlDeviceGetPgpuMetadataString(handle):
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetPgpuMetadataString')
    c_pgpuMetadata = create_string_buffer(NVML_VGPU_PGPU_METADATA_OPAQUE_DATA_SIZE)
    c_bufferSize = c_uint(0)
    ret = fn(handle, byref(c_pgpuMetadata), byref(c_bufferSize))
    if ret == NVML_ERROR_INSUFFICIENT_SIZE:
        ret = fn(handle, byref(c_pgpuMetadata), byref(c_bufferSize))
        _nvmlCheckReturn(ret)
    else:
        raise NVMLError(ret)
    return (c_pgpuMetadata.value, c_bufferSize.value)