from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlGpuInstanceGetComputeInstancePossiblePlacements(gpuInstance, profileId, placementsRef, countRef):
    fn = _nvmlGetFunctionPointer('nvmlGpuInstanceGetComputeInstancePossiblePlacements')
    ret = fn(gpuInstance, profileId, placementsRef, countRef)
    _nvmlCheckReturn(ret)
    return ret