import unittest
import os
from test.support import captured_stdout
from distutils.cmd import Command
from distutils.dist import Distribution
from distutils.errors import DistutilsOptionError
from distutils import debug
def test_ensure_string(self):
    cmd = self.cmd
    cmd.option1 = 'ok'
    cmd.ensure_string('option1')
    cmd.option2 = None
    cmd.ensure_string('option2', 'xxx')
    self.assertTrue(hasattr(cmd, 'option2'))
    cmd.option3 = 1
    self.assertRaises(DistutilsOptionError, cmd.ensure_string, 'option3')