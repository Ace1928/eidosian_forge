import os
import unittest
from distutils.core import PyPIRCCommand
from distutils.core import Distribution
from distutils.log import set_threshold
from distutils.log import WARN
from distutils.tests import support
def test_server_registration(self):
    self.write_file(self.rc, PYPIRC)
    cmd = self._cmd(self.dist)
    config = cmd._read_pypirc()
    config = list(sorted(config.items()))
    waited = [('password', 'secret'), ('realm', 'pypi'), ('repository', 'https://upload.pypi.org/legacy/'), ('server', 'server1'), ('username', 'me')]
    self.assertEqual(config, waited)
    self.write_file(self.rc, PYPIRC_OLD)
    config = cmd._read_pypirc()
    config = list(sorted(config.items()))
    waited = [('password', 'secret'), ('realm', 'pypi'), ('repository', 'https://upload.pypi.org/legacy/'), ('server', 'server-login'), ('username', 'tarek')]
    self.assertEqual(config, waited)