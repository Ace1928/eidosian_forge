from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlGpmQueryIfStreamingEnabled(device):
    c_state = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlGpmQueryIfStreamingEnabled')
    ret = fn(device, byref(c_state))
    _nvmlCheckReturn(ret)
    return c_state.value