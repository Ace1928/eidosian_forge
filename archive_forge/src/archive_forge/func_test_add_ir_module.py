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
def test_add_ir_module(self):
    lljit, rt_sum, cfunc_sum = self.jit()
    rt_mul = llvm.JITLibraryBuilder().add_ir(asm_mul.format(triple=llvm.get_default_triple())).export_symbol('mul').link(lljit, 'mul')
    res = CFUNCTYPE(c_int, c_int, c_int)(rt_mul['mul'])(2, -5)
    self.assertEqual(-10, res)
    self.assertNotEqual(lljit.lookup('sum', 'sum')['sum'], 0)
    self.assertNotEqual(lljit.lookup('mul', 'mul')['mul'], 0)
    with self.assertRaises(RuntimeError):
        lljit.lookup('sum', 'mul')
    with self.assertRaises(RuntimeError):
        lljit.lookup('mul', 'sum')