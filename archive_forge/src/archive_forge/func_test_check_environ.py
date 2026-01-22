import os
import sys
import unittest
from copy import copy
from unittest import mock
from distutils.errors import DistutilsPlatformError, DistutilsByteCompileError
from distutils.util import (get_platform, convert_path, change_root,
from distutils import util # used to patch _environ_checked
from distutils.sysconfig import get_config_vars
from distutils import sysconfig
from distutils.tests import support
import _osx_support
def test_check_environ(self):
    util._environ_checked = 0
    os.environ.pop('HOME', None)
    check_environ()
    self.assertEqual(os.environ['PLAT'], get_platform())
    self.assertEqual(util._environ_checked, 1)