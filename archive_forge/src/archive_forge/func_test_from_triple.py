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
def test_from_triple(self):
    f = llvm.Target.from_triple
    with self.assertRaises(RuntimeError) as cm:
        f('foobar')
    self.assertIn('No available targets are compatible with', str(cm.exception))
    triple = llvm.get_default_triple()
    target = f(triple)
    self.assertEqual(target.triple, triple)
    target.close()