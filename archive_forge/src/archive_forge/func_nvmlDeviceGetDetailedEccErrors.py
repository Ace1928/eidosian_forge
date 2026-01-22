from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetDetailedEccErrors(handle, errorType, counterType):
    c_counts = c_nvmlEccErrorCounts_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetDetailedEccErrors')
    ret = fn(handle, _nvmlMemoryErrorType_t(errorType), _nvmlEccCounterType_t(counterType), byref(c_counts))
    _nvmlCheckReturn(ret)
    return c_counts