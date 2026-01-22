from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlVgpuTypeGetMaxInstances(handle, vgpuTypeId):
    c_max_instances = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuTypeGetMaxInstances')
    ret = fn(handle, vgpuTypeId, byref(c_max_instances))
    _nvmlCheckReturn(ret)
    return c_max_instances.value