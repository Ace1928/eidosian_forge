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
def test_rv32d_ilp32f(self):
    self.check_riscv_target()
    llmod = self.fpadd_ll_module()
    target = self.riscv_target_machine(features='+f,+d', abiname='ilp32f')
    self.assertEqual(self.break_up_asm(target.emit_assembly(llmod)), riscv_asm_ilp32f)