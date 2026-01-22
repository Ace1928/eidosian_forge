from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetGpuInstances(device, profileId, gpuInstancesRef, countRef):
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetGpuInstances')
    ret = fn(device, profileId, gpuInstancesRef, countRef)
    _nvmlCheckReturn(ret)
    return ret