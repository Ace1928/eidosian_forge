from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceClearCpuAffinity(handle):
    fn = _nvmlGetFunctionPointer('nvmlDeviceClearCpuAffinity')
    ret = fn(handle)
    _nvmlCheckReturn(ret)
    return None