from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlVgpuTypeGetGpuInstanceProfileId(vgpuTypeId):
    c_profile_id = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuTypeGetGpuInstanceProfileId')
    ret = fn(vgpuTypeId, byref(c_profile_id))
    _nvmlCheckReturn(ret)
    return c_profile_id.value