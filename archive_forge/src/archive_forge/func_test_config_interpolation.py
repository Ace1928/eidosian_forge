import os
import unittest
from distutils.core import PyPIRCCommand
from distutils.core import Distribution
from distutils.log import set_threshold
from distutils.log import WARN
from distutils.tests import support
def test_config_interpolation(self):
    self.write_file(self.rc, PYPIRC)
    cmd = self._cmd(self.dist)
    cmd.repository = 'server3'
    config = cmd._read_pypirc()
    config = list(sorted(config.items()))
    waited = [('password', 'yh^%#rest-of-my-password'), ('realm', 'pypi'), ('repository', 'https://upload.pypi.org/legacy/'), ('server', 'server3'), ('username', 'cbiggles')]
    self.assertEqual(config, waited)