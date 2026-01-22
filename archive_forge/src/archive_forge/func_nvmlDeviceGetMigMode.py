from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetMigMode(device):
    c_currentMode = c_uint()
    c_pendingMode = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMigMode')
    ret = fn(device, byref(c_currentMode), byref(c_pendingMode))
    _nvmlCheckReturn(ret)
    return [c_currentMode.value, c_pendingMode.value]