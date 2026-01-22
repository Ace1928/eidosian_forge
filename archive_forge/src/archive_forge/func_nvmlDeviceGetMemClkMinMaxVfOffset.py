from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetMemClkMinMaxVfOffset(device, minOffset, maxOffset):
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetMemClkMinMaxVfOffset')
    ret = fn(device, minOffset, maxOffset)
    _nvmlCheckReturn(ret)
    return ret