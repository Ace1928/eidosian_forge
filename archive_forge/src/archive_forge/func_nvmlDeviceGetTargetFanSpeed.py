from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetTargetFanSpeed(handle, fan):
    c_speed = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetTargetFanSpeed')
    ret = fn(handle, fan, byref(c_speed))
    _nvmlCheckReturn(ret)
    return c_speed.value