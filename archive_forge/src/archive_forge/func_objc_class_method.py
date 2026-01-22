import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
def objc_class_method(objc_cls, objc_cmd, *args):
    py_cls = ObjCClass(objc_cls)
    py_cls.objc_cmd = objc_cmd
    args = convert_method_arguments(encoding, args)
    result = f(py_cls, *args)
    if isinstance(result, ObjCClass):
        result = result.ptr.value
    elif isinstance(result, ObjCInstance):
        result = result.ptr.value
    return result