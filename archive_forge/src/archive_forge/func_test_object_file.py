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
def test_object_file(self):
    target_machine = self.target_machine(jit=False)
    mod = self.module()
    obj_bin = target_machine.emit_object(mod)
    obj = llvm.ObjectFileRef.from_data(obj_bin)
    has_text = False
    last_address = -1
    for s in obj.sections():
        if s.is_text():
            has_text = True
            self.assertIsNotNone(s.name())
            self.assertTrue(s.size() > 0)
            self.assertTrue(len(s.data()) > 0)
            self.assertIsNotNone(s.address())
            self.assertTrue(last_address < s.address())
            last_address = s.address()
            break
    self.assertTrue(has_text)