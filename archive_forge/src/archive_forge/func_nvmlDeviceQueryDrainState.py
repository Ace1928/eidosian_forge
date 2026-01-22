from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceQueryDrainState(pciInfo):
    c_newState = c_uint()
    fn = _nvmlGetFunctionPointer('nvmlDeviceQueryDrainState')
    ret = fn(pointer(pciInfo), byref(c_newState))
    _nvmlCheckReturn(ret)
    return c_newState.value