from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetTotalEnergyConsumption(handle):
    c_millijoules = c_uint64()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetTotalEnergyConsumption')
    ret = fn(handle, byref(c_millijoules))
    _nvmlCheckReturn(ret)
    return c_millijoules.value