from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlDeviceGetFieldValues(handle, fieldIds):
    values_arr = c_nvmlFieldValue_t * len(fieldIds)
    values = values_arr()
    fn = _nvmlGetFunctionPointer('nvmlDeviceGetFieldValues')
    for i, fieldId in enumerate(fieldIds):
        try:
            values[i].fieldId, values[i].scopeId = fieldId
        except TypeError:
            values[i].fieldId = fieldId
    ret = fn(handle, c_int32(len(fieldIds)), byref(values))
    _nvmlCheckReturn(ret)
    return values