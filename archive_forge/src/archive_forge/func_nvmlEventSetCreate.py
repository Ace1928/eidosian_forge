from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlEventSetCreate():
    fn = _nvmlGetFunctionPointer('nvmlEventSetCreate')
    eventSet = c_nvmlEventSet_t()
    ret = fn(byref(eventSet))
    _nvmlCheckReturn(ret)
    return eventSet