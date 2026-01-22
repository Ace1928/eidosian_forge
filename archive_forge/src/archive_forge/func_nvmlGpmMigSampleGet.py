from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlGpmMigSampleGet(device, gpuInstanceId, gpmSample):
    fn = _nvmlGetFunctionPointer('nvmlGpmMigSampleGet')
    ret = fn(device, gpuInstanceId, gpmSample)
    _nvmlCheckReturn(ret)
    return gpmSample