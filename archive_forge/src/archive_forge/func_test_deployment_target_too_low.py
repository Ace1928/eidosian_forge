import sys
import os
from io import StringIO
import textwrap
from distutils.core import Distribution
from distutils.command.build_ext import build_ext
from distutils import sysconfig
from distutils.tests.support import (TempdirManager, LoggingSilencer,
from distutils.extension import Extension
from distutils.errors import (
import unittest
from test import support
from test.support import os_helper
from test.support.script_helper import assert_python_ok
from test.support import threading_helper
@unittest.skipUnless(sys.platform == 'darwin', 'test only relevant for MacOSX')
def test_deployment_target_too_low(self):
    self.assertRaises(DistutilsPlatformError, self._try_compile_deployment_target, '>', '10.1')