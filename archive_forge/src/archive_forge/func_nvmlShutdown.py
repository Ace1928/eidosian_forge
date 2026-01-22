from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlShutdown():
    fn = _nvmlGetFunctionPointer('nvmlShutdown')
    ret = fn()
    _nvmlCheckReturn(ret)
    global _nvmlLib_refcount
    libLoadLock.acquire()
    if 0 < _nvmlLib_refcount:
        _nvmlLib_refcount -= 1
    libLoadLock.release()
    return None