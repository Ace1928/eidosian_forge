from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlVgpuInstanceGetFBCStats(vgpuInstance):
    c_fbcStats = c_nvmlFBCStats_t()
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetFBCStats')
    ret = fn(vgpuInstance, byref(c_fbcStats))
    _nvmlCheckReturn(ret)
    return c_fbcStats