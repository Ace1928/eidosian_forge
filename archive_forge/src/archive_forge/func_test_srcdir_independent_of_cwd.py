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
def test_srcdir_independent_of_cwd(self):
    srcdir = sysconfig.get_config_var('srcdir')
    cwd = os.getcwd()
    try:
        os.chdir('..')
        srcdir2 = sysconfig.get_config_var('srcdir')
    finally:
        os.chdir(cwd)
    self.assertEqual(srcdir, srcdir2)