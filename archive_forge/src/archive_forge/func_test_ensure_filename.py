import unittest
import os
from test.support import captured_stdout
from distutils.cmd import Command
from distutils.dist import Distribution
from distutils.errors import DistutilsOptionError
from distutils import debug
def test_ensure_filename(self):
    cmd = self.cmd
    cmd.option1 = __file__
    cmd.ensure_filename('option1')
    cmd.option2 = 'xxx'
    self.assertRaises(DistutilsOptionError, cmd.ensure_filename, 'option2')