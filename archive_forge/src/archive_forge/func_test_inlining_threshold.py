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
def test_inlining_threshold(self):
    pmb = self.pmb()
    with self.assertRaises(NotImplementedError):
        pmb.inlining_threshold
    for i in (25, 80, 350):
        pmb.inlining_threshold = i