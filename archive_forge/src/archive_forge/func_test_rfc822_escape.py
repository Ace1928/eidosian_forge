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
def test_rfc822_escape(self):
    header = 'I am a\npoor\nlonesome\nheader\n'
    res = rfc822_escape(header)
    wanted = 'I am a%(8s)spoor%(8s)slonesome%(8s)sheader%(8s)s' % {'8s': '\n' + 8 * ' '}
    self.assertEqual(res, wanted)