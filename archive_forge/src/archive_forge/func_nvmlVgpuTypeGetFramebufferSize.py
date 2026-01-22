from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlVgpuTypeGetFramebufferSize(vgpuTypeId):
    c_fb_size = c_ulonglong(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuTypeGetFramebufferSize')
    ret = fn(vgpuTypeId, byref(c_fb_size))
    _nvmlCheckReturn(ret)
    return c_fb_size.value