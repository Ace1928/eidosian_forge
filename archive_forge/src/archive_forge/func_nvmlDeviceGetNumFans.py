from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetNumFans(device):
    c_numFans = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetNumFans')
    ret = fn(device, byref(c_numFans))
    _nvmlCheckReturn(ret)
    return c_numFans.value