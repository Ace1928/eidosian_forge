from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetThermalSettings(device, sensorindex, c_thermalsettings):
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetThermalSettings')
    ret = fn(device, sensorindex, c_thermalsettings)
    _nvmlCheckReturn(ret)
    return ret