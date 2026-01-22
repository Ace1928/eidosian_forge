import contextlib
import ctypes
import struct
import sys
import llvmlite.ir as ir
import numpy as np
import unittest
from numba.core import types, typing, cgutils, cpu
from numba.core.compiler_lock import global_compiler_lock
from numba.tests.support import TestCase, run_in_subprocess
def get_bytearray_addr(self, ba):
    assert isinstance(ba, bytearray)
    ba_as_string = ctypes.pythonapi.PyByteArray_AsString
    ba_as_string.argtypes = [ctypes.py_object]
    ba_as_string.restype = ctypes.c_void_p
    return ba_as_string(ba)