from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlVgpuTypeGetMaxInstancesPerVm(vgpuTypeId):
    c_max_instances_per_vm = c_uint(0)
    fn = _nvmlGetFunctionPointer('nvmlVgpuTypeGetMaxInstancesPerVm')
    ret = fn(vgpuTypeId, byref(c_max_instances_per_vm))
    _nvmlCheckReturn(ret)
    return c_max_instances_per_vm.value