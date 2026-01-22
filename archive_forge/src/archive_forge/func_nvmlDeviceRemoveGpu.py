from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceRemoveGpu(pciInfo):
    fn = _nvmlGetFunctionPointer('nvmlDeviceRemoveGpu')
    ret = fn(pointer(pciInfo))
    _nvmlCheckReturn(ret)
    return None