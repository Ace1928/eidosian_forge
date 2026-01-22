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
def test_lookup_undefined_symbol_fails(self):
    lljit = llvm.create_lljit_compiler()
    with self.assertRaisesRegex(RuntimeError, 'No such library'):
        lljit.lookup('foo', '__foobar')
    rt = llvm.JITLibraryBuilder().import_symbol('__xyzzy', 1234).export_symbol('__xyzzy').link(lljit, 'foo')
    self.assertNotEqual(rt['__xyzzy'], 0)
    with self.assertRaisesRegex(RuntimeError, 'Symbols not found.*__foobar'):
        lljit.lookup('foo', '__foobar')