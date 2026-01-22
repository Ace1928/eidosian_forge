import logging
import re
import sys
import warnings
from ctypes import (c_void_p, c_int, POINTER, c_char_p, c_size_t, byref,
import threading
from llvmlite import ir
from .error import NvvmError, NvvmSupportError, NvvmWarning
from .libs import get_libdevice, open_libdevice, open_cudalib
from numba.core import cgutils, config
def get_version(self):
    major = c_int()
    minor = c_int()
    err = self.nvvmVersion(byref(major), byref(minor))
    self.check_error(err, 'Failed to get version.')
    return (major.value, minor.value)