from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetDynamicPstatesInfo(device, c_dynamicpstatesinfo):
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetDynamicPstatesInfo')
    ret = fn(device, c_dynamicpstatesinfo)
    _nvmlCheckReturn(ret)
    return ret