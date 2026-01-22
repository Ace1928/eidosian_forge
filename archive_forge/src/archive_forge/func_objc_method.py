import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
def objc_method(objc_self, objc_cmd, *args):
    py_self = ObjCInstance(objc_self, True)
    py_self.objc_cmd = objc_cmd
    py_self.retained = True
    args = convert_method_arguments(encoding, args)
    result = f(py_self, *args)
    if isinstance(result, ObjCClass):
        result = result.ptr.value
    elif isinstance(result, ObjCInstance):
        result = result.ptr.value
    return result