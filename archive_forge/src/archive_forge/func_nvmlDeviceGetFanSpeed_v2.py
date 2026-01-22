from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetFanSpeed_v2(handle, fan):
    c_speed = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetFanSpeed_v2')
    ret = fn(handle, fan, byref(c_speed))
    _nvmlCheckReturn(ret)
    return c_speed.value