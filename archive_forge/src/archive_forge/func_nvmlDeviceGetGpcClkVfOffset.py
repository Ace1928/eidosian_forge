from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetGpcClkVfOffset(device):
    offset = c_int32()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetGpcClkVfOffset')
    ret = fn(device, byref(offset))
    _nvmlCheckReturn(ret)
    return offset.value