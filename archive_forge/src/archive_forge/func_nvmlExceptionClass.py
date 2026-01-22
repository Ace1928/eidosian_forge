from ctypes import *
from ctypes.util import find_library
from functools import wraps
import sys
import os
import threading
import string
def nvmlExceptionClass(nvmlErrorCode):
    if nvmlErrorCode not in NVMLError._valClassMapping:
        raise ValueError('nvmlErrorCode %s is not valid' % nvmlErrorCode)
    return NVMLError._valClassMapping[nvmlErrorCode]