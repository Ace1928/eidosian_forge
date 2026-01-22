from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlGpmSampleAlloc():
    gpmSample = c_nvmlGpmSample_t()
    fn = _nvmlGetFunctionPointer('nvmlGpmSampleAlloc')
    ret = fn(byref(gpmSample))
    _nvmlCheckReturn(ret)
    return gpmSample