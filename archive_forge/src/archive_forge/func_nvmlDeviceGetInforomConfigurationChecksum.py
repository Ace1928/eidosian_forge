from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetInforomConfigurationChecksum(handle):
    c_checksum = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetInforomConfigurationChecksum')
    ret = fn(handle, byref(c_checksum))
    _nvmlCheckReturn(ret)
    return c_checksum.value