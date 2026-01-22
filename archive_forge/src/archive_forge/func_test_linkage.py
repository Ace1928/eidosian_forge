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
def test_linkage(self):
    mod = self.module()
    glob = mod.get_global_variable('glob')
    linkage = glob.linkage
    self.assertIsInstance(glob.linkage, llvm.Linkage)
    glob.linkage = linkage
    self.assertEqual(glob.linkage, linkage)
    for linkage in ('internal', 'external'):
        glob.linkage = linkage
        self.assertIsInstance(glob.linkage, llvm.Linkage)
        self.assertEqual(glob.linkage.name, linkage)