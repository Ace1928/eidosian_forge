from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetTopologyCommonAncestor(device1, device2):
    c_level = _nvmlGpuTopologyLevel_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetTopologyCommonAncestor')
    ret = fn(device1, device2, byref(c_level))
    _nvmlCheckReturn(ret)
    return c_level.value