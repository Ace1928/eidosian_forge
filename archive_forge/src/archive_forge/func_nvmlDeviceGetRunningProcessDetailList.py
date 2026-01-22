from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetRunningProcessDetailList(handle, version, mode):
    c_processDetailList = c_nvmlProcessDetailList_t()
    c_processDetailList.version = version
    c_processDetailList.mode = mode
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetRunningProcessDetailList')
    ret = fn(handle, c_processDetailList)
    return ret