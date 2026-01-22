from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceSetConfComputeUnprotectedMemSize(device, c_ccMemSize):
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetConfComputeUnprotectedMemSize')
    ret = fn(device, c_ccMemSize)
    _nvmlCheckReturn(ret)
    return ret