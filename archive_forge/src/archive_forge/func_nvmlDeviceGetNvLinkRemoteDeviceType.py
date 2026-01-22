from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetNvLinkRemoteDeviceType(handle, link):
    c_type = _nvmlNvLinkDeviceType_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetNvLinkRemoteDeviceType')
    ret = fn(handle, link, byref(c_type))
    _nvmlCheckReturn(ret)
    return c_type.value