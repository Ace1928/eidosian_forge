import os
import unittest
from distutils.core import PyPIRCCommand
from distutils.core import Distribution
from distutils.log import set_threshold
from distutils.log import WARN
from distutils.tests import support
def test_server_empty_registration(self):
    cmd = self._cmd(self.dist)
    rc = cmd._get_rc_file()
    self.assertFalse(os.path.exists(rc))
    cmd._store_pypirc('tarek', 'xxx')
    self.assertTrue(os.path.exists(rc))
    f = open(rc)
    try:
        content = f.read()
        self.assertEqual(content, WANTED)
    finally:
        f.close()