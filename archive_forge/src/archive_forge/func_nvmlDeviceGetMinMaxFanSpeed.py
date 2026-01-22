from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetMinMaxFanSpeed(handle, minSpeed, maxSpeed):
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMinMaxFanSpeed')
    ret = fn(handle, minSpeed, maxSpeed)
    _nvmlCheckReturn(ret)
    return ret