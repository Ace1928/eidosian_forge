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
def test_add_object_file_from_filesystem(self):
    target_machine = self.target_machine(jit=False)
    mod = self.module()
    obj_bin = target_machine.emit_object(mod)
    temp_desc, temp_path = mkstemp()
    try:
        try:
            f = os.fdopen(temp_desc, 'wb')
            f.write(obj_bin)
            f.flush()
        finally:
            f.close()
        jit = llvm.create_mcjit_compiler(self.module(self.mod_asm), target_machine)
        jit.add_object_file(temp_path)
    finally:
        os.unlink(temp_path)
    sum_twice = CFUNCTYPE(c_int, c_int, c_int)(jit.get_function_address('sum_twice'))
    self.assertEqual(sum_twice(2, 3), 10)