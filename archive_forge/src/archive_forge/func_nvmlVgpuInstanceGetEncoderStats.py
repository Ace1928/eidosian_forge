from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlVgpuInstanceGetEncoderStats(vgpuInstance):
    c_encoderCount = c_ulonglong(0)
    c_encodeFps = c_ulonglong(0)
    c_encoderLatency = c_ulonglong(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetEncoderStats')
    ret = fn(vgpuInstance, byref(c_encoderCount), byref(c_encodeFps), byref(c_encoderLatency))
    _nvmlCheckReturn(ret)
    return (c_encoderCount.value, c_encodeFps.value, c_encoderLatency.value)