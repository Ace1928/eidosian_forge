import ctypes
import threading
from ctypes import CFUNCTYPE, c_int, c_int32
from ctypes.util import find_library
import gc
import locale
import os
import platform
import re
import subprocess
import sys
import unittest
from contextlib import contextmanager
from tempfile import mkstemp
from llvmlite import ir
from llvmlite import binding as llvm
from llvmlite.binding import ffi
from llvmlite.tests import TestCase
def test_global_ctors_dtors(self):
    mod = self.module(asm_global_ctors)
    ee = self.jit(mod)
    ee.finalize_object()
    ee.run_static_constructors()
    ptr_addr = ee.get_global_value_address('A')
    ptr_t = ctypes.POINTER(ctypes.c_int32)
    ptr = ctypes.cast(ptr_addr, ptr_t)
    self.assertEqual(ptr.contents.value, 10)
    foo_addr = ee.get_function_address('foo')
    foo = ctypes.CFUNCTYPE(ctypes.c_int32)(foo_addr)
    self.assertEqual(foo(), 12)
    ee.run_static_destructors()
    self.assertEqual(ptr.contents.value, 20)