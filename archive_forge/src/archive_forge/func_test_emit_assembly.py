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
def test_emit_assembly(self):
    """Test TargetMachineRef.emit_assembly()"""
    target_machine = self.target_machine(jit=True)
    mod = self.module()
    ee = self.jit(mod, target_machine)
    raw_asm = target_machine.emit_assembly(mod)
    self.assertIn('sum', raw_asm)
    target_machine.set_asm_verbosity(True)
    raw_asm_verbose = target_machine.emit_assembly(mod)
    self.assertIn('sum', raw_asm)
    self.assertNotEqual(raw_asm, raw_asm_verbose)