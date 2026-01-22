from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetNvLinkUtilizationCounter(device, link, counter):
    c_rxcounter = c_ulonglong()
    c_txcounter = c_ulonglong()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetNvLinkUtilizationCounter')
    ret = fn(device, link, counter, byref(c_rxcounter), byref(c_txcounter))
    _nvmlCheckReturn(ret)
    return (c_rxcounter.value, c_txcounter.value)