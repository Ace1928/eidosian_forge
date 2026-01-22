from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlUnitGetHandleByIndex(index):
    c_index = c_uint(index)
    unit = c_nvmlUnit_t()
    fn = _nvmlGetFunctionPointer('nvmlUnitGetHandleByIndex')
    ret = fn(c_index, byref(unit))
    _nvmlCheckReturn(ret)
    return unit