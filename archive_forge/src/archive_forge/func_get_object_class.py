import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
def get_object_class(obj):
    return c_void_p(objc.object_getClass(obj))