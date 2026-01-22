import io
import distutils.core
import os
import shutil
import sys
from test.support import captured_stdout
from test.support import os_helper
import unittest
from distutils.tests import support
from distutils import log
from distutils.core import setup
import os
from distutils.core import setup
from distutils.core import setup
from distutils.core import setup
from distutils.command.install import install as _install
def test_debug_mode(self):
    sys.argv = ['setup.py', '--name']
    with captured_stdout() as stdout:
        distutils.core.setup(name='bar')
    stdout.seek(0)
    self.assertEqual(stdout.read(), 'bar\n')
    distutils.core.DEBUG = True
    try:
        with captured_stdout() as stdout:
            distutils.core.setup(name='bar')
    finally:
        distutils.core.DEBUG = False
    stdout.seek(0)
    wanted = 'options (after parsing config files):\n'
    self.assertEqual(stdout.readlines()[0], wanted)