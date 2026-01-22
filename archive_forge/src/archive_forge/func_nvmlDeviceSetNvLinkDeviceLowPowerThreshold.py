from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceSetNvLinkDeviceLowPowerThreshold(device, l1threshold):
    c_info = c_nvmlNvLinkPowerThres_t()
    c_info.lowPwrThreshold = l1threshold
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetNvLinkDeviceLowPowerThreshold')
    ret = fn(device, byref(c_info))
    _nvmlCheckReturn(ret)
    return ret