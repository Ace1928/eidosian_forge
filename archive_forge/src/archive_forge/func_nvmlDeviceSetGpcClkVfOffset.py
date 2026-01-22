from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceSetGpcClkVfOffset(device, offset):
    c_offset = c_int32(offset)
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetGpcClkVfOffset')
    ret = fn(device, c_offset)
    _nvmlCheckReturn(ret)
    return ret