from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetVgpuSchedulerState(handle):
    c_vgpu_sched_state = c_nvmlVgpuSchedulerGetState_t()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetVgpuSchedulerState')
    ret = fn(handle, byref(c_vgpu_sched_state))
    _nvmlCheckReturn(ret)
    return c_vgpu_sched_state