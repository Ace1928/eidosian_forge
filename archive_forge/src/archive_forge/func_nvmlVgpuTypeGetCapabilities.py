from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlVgpuTypeGetCapabilities(vgpuTypeId, capability):
    c_cap_result = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuTypeGetCapabilities')
    ret = fn(vgpuTypeId, _nvmlVgpuCapability_t(capability), byref(c_cap_result))
    _nvmlCheckReturn(ret)
    return c_cap_result.value