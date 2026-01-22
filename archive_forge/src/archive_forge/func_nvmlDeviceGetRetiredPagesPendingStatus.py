from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetRetiredPagesPendingStatus(device):
    c_pending = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetRetiredPagesPendingStatus')
    ret = fn(device, byref(c_pending))
    _nvmlCheckReturn(ret)
    return int(c_pending.value)