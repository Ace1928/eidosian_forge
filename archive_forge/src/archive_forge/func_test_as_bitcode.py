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
def test_as_bitcode(self):
    mod = self.module()
    bc = mod.as_bitcode()
    bitcode_wrapper_magic = b'\xde\xc0\x17\x0b'
    bitcode_magic = b'BC'
    self.assertTrue(bc.startswith(bitcode_magic) or bc.startswith(bitcode_wrapper_magic))