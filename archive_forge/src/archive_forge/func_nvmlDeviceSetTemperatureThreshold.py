from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceSetTemperatureThreshold(handle, threshold, temp):
    c_temp = c_uint()
    c_temp.value = temp
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetTemperatureThreshold')
    ret = fn(handle, _nvmlTemperatureThresholds_t(threshold), byref(c_temp))
    _nvmlCheckReturn(ret)
    return None