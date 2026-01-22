from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetConfComputeGpuAttestationReport(device, c_nonce):
    c_attestReport = c_nvmlConfComputeGpuAttestationReport_t()
    c_nonce_arr = (c_uint8 * len(c_nonce))(*c_nonce)
    setattr(c_attestReport, 'nonce', c_nonce_arr)
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetConfComputeGpuAttestationReport')
    ret = fn(device, byref(c_attestReport))
    _nvmlCheckReturn(ret)
    return c_attestReport