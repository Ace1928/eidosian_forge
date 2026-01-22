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
@unittest.skipUnless(platform.machine().startswith('x86'), 'x86 only')
def test_target_library_info_behavior(self):
    """Test a specific situation that demonstrate TLI is affecting
        optimization. See https://github.com/numba/numba/issues/8898.
        """

    def run(use_tli):
        mod = llvm.parse_assembly(asm_tli_exp2)
        target = llvm.Target.from_triple(mod.triple)
        tm = target.create_target_machine()
        pm = llvm.ModulePassManager()
        tm.add_analysis_passes(pm)
        if use_tli:
            pm.add_target_library_info(mod.triple)
        pm.add_instruction_combining_pass()
        pm.run(mod)
        return mod
    mod = run(use_tli=True)
    self.assertIn('call float @llvm.exp2.f32', str(mod))
    mod = run(use_tli=False)
    self.assertNotIn('call float @llvm.exp2.f32', str(mod))
    self.assertIn('call float @ldexpf', str(mod))