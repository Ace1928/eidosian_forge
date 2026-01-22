import contextlib
import os
import shutil
import subprocess
import sys
import textwrap
import unittest
from distutils import sysconfig
from distutils.ccompiler import get_default_compiler
from distutils.tests import support
from test.support import swap_item, requires_subprocess, is_wasi
from test.support.os_helper import TESTFN
from test.support.warnings_helper import check_warnings
@unittest.skipIf(is_wasi, 'Incompatible with WASI mapdir and OOT builds')
def test_srcdir(self):
    srcdir = sysconfig.get_config_var('srcdir')
    self.assertTrue(os.path.isabs(srcdir), srcdir)
    self.assertTrue(os.path.isdir(srcdir), srcdir)
    if sysconfig.python_build:
        Python_h = os.path.join(srcdir, 'Include', 'Python.h')
        self.assertTrue(os.path.exists(Python_h), Python_h)
        pyconfig_h = os.path.join(srcdir, 'PC', 'pyconfig.h')
        self.assertTrue(os.path.exists(pyconfig_h), pyconfig_h)
        pyconfig_h_in = os.path.join(srcdir, 'pyconfig.h.in')
        self.assertTrue(os.path.exists(pyconfig_h_in), pyconfig_h_in)
    elif os.name == 'posix':
        self.assertEqual(os.path.dirname(sysconfig.get_makefile_filename()), srcdir)