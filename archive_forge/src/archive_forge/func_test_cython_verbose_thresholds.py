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
def test_cython_verbose_thresholds(self):

    @contextmanager
    def mock_distutils():

        class MockLog:
            DEBUG = 1
            INFO = 2
            thresholds = [INFO]

            def set_threshold(self, val):
                self.thresholds.append(val)
                return self.thresholds[-2]
        new_log = MockLog()
        old_log = IpythonMagic.distutils.log
        try:
            IpythonMagic.distutils.log = new_log
            yield new_log
        finally:
            IpythonMagic.distutils.log = old_log
    ip = self._ip
    with mock_distutils() as verbose_log:
        ip.run_cell_magic('cython', '--verbose', code)
        ip.ex('g = f(10)')
    self.assertEqual(ip.user_ns['g'], 20.0)
    self.assertEqual([verbose_log.INFO, verbose_log.DEBUG, verbose_log.INFO], verbose_log.thresholds)
    with mock_distutils() as normal_log:
        ip.run_cell_magic('cython', '', code)
        ip.ex('g = f(10)')
    self.assertEqual(ip.user_ns['g'], 20.0)
    self.assertEqual([normal_log.INFO], normal_log.thresholds)