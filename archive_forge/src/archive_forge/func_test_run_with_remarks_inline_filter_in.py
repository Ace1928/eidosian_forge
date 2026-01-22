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
def test_run_with_remarks_inline_filter_in(self):
    pm = self.pm()
    pm.add_function_inlining_pass(70)
    self.pmb().populate(pm)
    mod = self.module(asm_inlineasm2)
    status, remarks = pm.run_with_remarks(mod, remarks_filter='inlin.*')
    self.assertTrue(status)
    self.assertIn('Passed', remarks)
    self.assertIn('inlineme', remarks)