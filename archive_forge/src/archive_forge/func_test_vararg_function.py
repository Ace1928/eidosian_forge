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
def test_vararg_function(self):
    mod = self.module(asm_vararg_declare)
    func = mod.get_function('vararg')
    decltype = func.type.element_type
    self.assertTrue(decltype.is_function_vararg)
    mod = self.module(asm_sum_declare)
    func = mod.get_function('sum')
    decltype = func.type.element_type
    self.assertFalse(decltype.is_function_vararg)
    self.assertTrue(func.type.is_pointer)
    with self.assertRaises(ValueError) as raises:
        func.type.is_function_vararg
    self.assertIn('Type i32 (i32, i32)* is not a function', str(raises.exception))