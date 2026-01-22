from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceSetDefaultAutoBoostedClocksEnabled(handle, enabled, flags):
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetDefaultAutoBoostedClocksEnabled')
    ret = fn(handle, _nvmlEnableState_t(enabled), c_uint(flags))
    _nvmlCheckReturn(ret)
    return None