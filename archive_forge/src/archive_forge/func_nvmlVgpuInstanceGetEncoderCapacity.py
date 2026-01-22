from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlVgpuInstanceGetEncoderCapacity(vgpuInstance):
    c_encoder_capacity = c_ulonglong(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetEncoderCapacity')
    ret = fn(vgpuInstance, byref(c_encoder_capacity))
    _nvmlCheckReturn(ret)
    return c_encoder_capacity.value