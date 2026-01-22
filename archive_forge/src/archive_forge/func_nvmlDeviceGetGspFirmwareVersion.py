from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetGspFirmwareVersion(handle, version):
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetGspFirmwareVersion')
    ret = fn(handle, version)
    _nvmlCheckReturn(ret)
    return ret