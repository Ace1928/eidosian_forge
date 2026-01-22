from __future__ import absolute_import
import os
import io
import sys
from contextlib import contextmanager
from unittest import skipIf
from Cython.Build import IpythonMagic
from Cython.TestUtils import CythonTest
from Cython.Compiler.Annotate import AnnotationCCodeWriter
from libc.math cimport sin
def test_cython2(self):
    ip = self._ip
    ip.run_cell_magic('cython', '-2', cython3_code)
    ip.ex('g = f(10); h = call(10)')
    self.assertEqual(ip.user_ns['g'], 2 // 10)
    self.assertEqual(ip.user_ns['h'], 2 // 10)