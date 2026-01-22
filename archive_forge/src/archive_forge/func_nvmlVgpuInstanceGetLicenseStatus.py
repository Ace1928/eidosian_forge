from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlVgpuInstanceGetLicenseStatus(vgpuInstance):
    c_license_status = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuInstanceGetLicenseStatus')
    ret = fn(vgpuInstance, byref(c_license_status))
    _nvmlCheckReturn(ret)
    return c_license_status.value