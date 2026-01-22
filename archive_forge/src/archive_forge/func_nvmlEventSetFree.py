from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlEventSetFree(eventSet):
    fn = _nvmlGetFunctionPointer('nvmlEventSetFree')
    ret = fn(eventSet)
    _nvmlCheckReturn(ret)
    return None