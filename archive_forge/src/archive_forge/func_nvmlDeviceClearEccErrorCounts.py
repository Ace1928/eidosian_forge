from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceClearEccErrorCounts(handle, counterType):
    fn = _nvmlGetFunctionPointer('nvmlDeviceClearEccErrorCounts')
    ret = fn(handle, _nvmlEccCounterType_t(counterType))
    _nvmlCheckReturn(ret)
    return None