from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceFreezeNvLinkUtilizationCounter(device, link, counter, freeze):
    fn = _nvmlGetFunctionPointer('nvmlDeviceFreezeNvLinkUtilizationCounter')
    ret = fn(device, link, counter, freeze)
    _nvmlCheckReturn(ret)
    return None