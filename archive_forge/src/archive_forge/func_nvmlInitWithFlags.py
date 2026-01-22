from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlInitWithFlags(flags):
    _LoadNvmlLibrary()
    fn = _nvmlGetFunctionPointer('nvmlInitWithFlags')
    ret = fn(flags)
    _nvmlCheckReturn(ret)
    global _nvmlLib_refcount
    libLoadLock.acquire()
    _nvmlLib_refcount += 1
    libLoadLock.release()
    return None