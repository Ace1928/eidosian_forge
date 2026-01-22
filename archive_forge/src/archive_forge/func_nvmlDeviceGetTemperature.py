from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetTemperature(handle, sensor):
    c_temp = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetTemperature')
    ret = fn(handle, _nvmlTemperatureSensors_t(sensor), byref(c_temp))
    _nvmlCheckReturn(ret)
    return c_temp.value