from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceSetPowerManagementLimit_v2(device, powerScope, powerLimit, version=nvmlPowerValue_v2):
    c_powerScope = _nvmlPowerScopeType_t(powerScope)
    c_powerValue = c_nvmlPowerValue_v2_t()
    c_powerValue.version = c_uint(version)
    c_powerValue.powerScope = c_powerScope
    c_powerValue.powerValueMw = c_uint(powerLimit)
    fn = _nvmlGetFunctionPointer('nvmlDeviceSetPowerManagementLimit_v2')
    ret = fn(device, byref(c_powerValue))
    return ret