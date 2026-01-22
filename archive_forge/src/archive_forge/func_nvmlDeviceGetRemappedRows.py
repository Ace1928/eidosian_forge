from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetRemappedRows(device):
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetRemappedRows')
    c_corr = c_uint()
    c_unc = c_uint()
    c_bpending = c_uint()
    c_bfailure = c_uint()
    ret = fn(device, byref(c_corr), byref(c_unc), byref(c_bpending), byref(c_bfailure))
    _nvmlCheckReturn(ret)
    return (c_corr.value, c_unc.value, c_bpending.value, c_bfailure.value)