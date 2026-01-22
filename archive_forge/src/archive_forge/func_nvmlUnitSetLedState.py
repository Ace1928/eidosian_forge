from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlUnitSetLedState(unit, color):
    fn = _nvmlGetFunctionPointer('nvmlUnitSetLedState')
    ret = fn(unit, _nvmlLedColor_t(color))
    _nvmlCheckReturn(ret)
    return None