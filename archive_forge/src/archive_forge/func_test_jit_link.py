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
def test_jit_link(self):
    if sys.platform == 'win32':
        with self.assertRaisesRegex(RuntimeError, 'JITLink .* Windows'):
            llvm.create_lljit_compiler(use_jit_link=True)
    else:
        self.assertIsNotNone(llvm.create_lljit_compiler(use_jit_link=True))