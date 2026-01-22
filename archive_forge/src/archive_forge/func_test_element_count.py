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
def test_element_count(self):
    mod = self.module()
    glob_struct_type = mod.get_struct_type('struct.glob_type')
    _, array_type = glob_struct_type.elements
    self.assertEqual(array_type.element_count, 2)
    with self.assertRaises(ValueError):
        glob_struct_type.element_count