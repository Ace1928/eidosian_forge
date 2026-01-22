from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlSetVgpuVersion(vgpuVersion):
    fn = _nvmlGetFunctionPointer('nvmlSetVgpuVersion')
    ret = fn(byref(vgpuVersion))
    _nvmlCheckReturn(ret)
    return ret