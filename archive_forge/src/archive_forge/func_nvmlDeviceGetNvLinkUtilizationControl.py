from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetNvLinkUtilizationControl(device, link, counter):
    c_control = nvmlNvLinkUtilizationControl_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetNvLinkUtilizationControl')
    ret = fn(device, link, counter, byref(c_control))
    _nvmlCheckReturn(ret)
    return c_control