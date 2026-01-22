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
@skip_win32
def test_extlibs(self):
    ip = self._ip
    code = u'\nfrom libc.math cimport sin\nx = sin(0.0)\n        '
    ip.user_ns['x'] = 1
    ip.run_cell_magic('cython', '-l m', code)
    self.assertEqual(ip.user_ns['x'], 0)