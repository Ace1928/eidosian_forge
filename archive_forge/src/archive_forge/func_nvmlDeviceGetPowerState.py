from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetPowerState(handle):
    c_pstate = _nvmlPstates_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetPowerState')
    ret = fn(handle, byref(c_pstate))
    _nvmlCheckReturn(ret)
    return c_pstate.value