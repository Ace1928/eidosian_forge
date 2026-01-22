from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlUnitGetPsuInfo(unit):
    c_info = c_nvmlPSUInfo_t()
    fn = _nvmlGetFunctionPointer('nvmlUnitGetPsuInfo')
    ret = fn(unit, byref(c_info))
    _nvmlCheckReturn(ret)
    return c_info