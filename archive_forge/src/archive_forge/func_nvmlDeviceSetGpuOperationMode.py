from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceSetGpuOperationMode(handle, mode):
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetGpuOperationMode')
    ret = fn(handle, _nvmlGpuOperationMode_t(mode))
    _nvmlCheckReturn(ret)
    return None