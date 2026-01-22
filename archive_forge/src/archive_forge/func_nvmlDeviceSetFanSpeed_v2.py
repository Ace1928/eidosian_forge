from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceSetFanSpeed_v2(handle, index, speed):
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetFanSpeed_v2')
    ret = fn(handle, index, speed)
    _nvmlCheckReturn(ret)
    return ret