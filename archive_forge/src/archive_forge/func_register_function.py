from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
def register_function(lib, item, ignore_errors):
    try:
        func = getattr(lib, item[0])
    except AttributeError as e:
        msg = str(e) + '. Please ensure that your python bindings are compatible with your libclang.so version.'
        if ignore_errors:
            return
        raise LibclangError(msg)
    if len(item) >= 2:
        func.argtypes = item[1]
    if len(item) >= 3:
        func.restype = item[2]
    if len(item) == 4:
        func.errcheck = item[3]