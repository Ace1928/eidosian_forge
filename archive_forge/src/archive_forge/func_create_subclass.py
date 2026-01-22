import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
def create_subclass(superclass, name):
    if isinstance(superclass, str):
        superclass = get_class(superclass)
    return c_void_p(objc.objc_allocateClassPair(superclass, ensure_bytes(name), 0))