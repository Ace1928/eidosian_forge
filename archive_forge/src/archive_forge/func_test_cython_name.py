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
def test_cython_name(self):
    ip = self._ip
    ip.run_cell_magic('cython', '--name=mymodule', code)
    ip.ex('import mymodule; g = mymodule.f(10)')
    self.assertEqual(ip.user_ns['g'], 20.0)