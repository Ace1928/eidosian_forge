from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
@convertStrBytes
def nvmlDeviceGetHandleByUUID(uuid):
    c_uuid = c_char_p(uuid)
    device = c_nvmlDevice_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetHandleByUUID')
    ret = fn(c_uuid, byref(device))
    _nvmlCheckReturn(ret)
    return device