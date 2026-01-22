from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlGetVgpuVersion(supported, current):
    fn = _nvmlGetFunctionPointer('nvmlGetVgpuVersion')
    ret = fn(byref(supported), byref(current))
    _nvmlCheckReturn(ret)
    return ret