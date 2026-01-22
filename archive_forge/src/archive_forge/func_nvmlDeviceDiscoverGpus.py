from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceDiscoverGpus(pciInfo):
    fn = _nvmlGetFunctionPointer('nvmlDeviceDiscoverGpus')
    ret = fn(pointer(pciInfo))
    _nvmlCheckReturn(ret)
    return None