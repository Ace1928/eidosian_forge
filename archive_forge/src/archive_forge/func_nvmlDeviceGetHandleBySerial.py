from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
@convertStrBytes
def nvmlDeviceGetHandleBySerial(serial):
    c_serial = c_char_p(serial)
    device = c_nvmlDevice_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetHandleBySerial')
    ret = fn(c_serial, byref(device))
    _nvmlCheckReturn(ret)
    return device