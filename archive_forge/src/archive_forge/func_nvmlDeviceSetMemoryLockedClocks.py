from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceSetMemoryLockedClocks(handle, minMemClockMHz, maxMemClockMHz):
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetMemoryLockedClocks')
    ret = fn(handle, c_uint(minMemClockMHz), c_uint(maxMemClockMHz))
    _nvmlCheckReturn(ret)
    return None