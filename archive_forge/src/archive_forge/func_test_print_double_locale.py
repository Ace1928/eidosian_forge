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
@unittest.skipIf(no_de_locale(), 'Locale not available')
def test_print_double_locale(self):
    m = self.module(asm_double_locale)
    expect = str(m)
    locale.setlocale(locale.LC_ALL, 'de_DE')
    got = str(m)
    self.assertEqual(expect, got)