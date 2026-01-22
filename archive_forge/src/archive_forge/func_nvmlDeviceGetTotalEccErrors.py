from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetTotalEccErrors(handle, errorType, counterType):
    c_count = c_ulonglong()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetTotalEccErrors')
    ret = fn(handle, _nvmlMemoryErrorType_t(errorType), _nvmlEccCounterType_t(counterType), byref(c_count))
    _nvmlCheckReturn(ret)
    return c_count.value