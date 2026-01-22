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
def test_dylib_symbols(self):
    llvm.add_symbol('__xyzzy', 1234)
    llvm.add_symbol('__xyzzy', 5678)
    addr = llvm.address_of_symbol('__xyzzy')
    self.assertEqual(addr, 5678)
    addr = llvm.address_of_symbol('__foobar')
    self.assertIs(addr, None)