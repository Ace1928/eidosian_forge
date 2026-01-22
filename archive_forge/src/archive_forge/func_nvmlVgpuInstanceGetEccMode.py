from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlVgpuInstanceGetEccMode(vgpuInstance):
    c_mode = _nvmlEnableState_t()
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetEccMode')
    ret = fn(vgpuInstance, byref(c_mode))
    _nvmlCheckReturn(ret)
    return c_mode.value