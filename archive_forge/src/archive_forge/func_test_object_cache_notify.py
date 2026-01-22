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
def test_object_cache_notify(self):
    notifies = []

    def notify(mod, buf):
        notifies.append((mod, buf))
    mod = self.module()
    ee = self.jit(mod)
    ee.set_object_cache(notify)
    self.assertEqual(len(notifies), 0)
    cfunc = self.get_sum(ee)
    cfunc(2, -5)
    self.assertEqual(len(notifies), 1)
    self.assertIs(notifies[0][0], mod)
    self.assertIsInstance(notifies[0][1], bytes)
    notifies[:] = []
    mod2 = self.module(asm_mul)
    ee.add_module(mod2)
    cfunc = self.get_sum(ee, 'mul')
    self.assertEqual(len(notifies), 1)
    self.assertIs(notifies[0][0], mod2)
    self.assertIsInstance(notifies[0][1], bytes)