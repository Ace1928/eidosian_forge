from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceResetNvLinkErrorCounters(device, link):
    fn = _nvmlGetFunctionPointer('nvmlDeviceResetNvLinkErrorCounters')
    ret = fn(device, link)
    _nvmlCheckReturn(ret)
    return None