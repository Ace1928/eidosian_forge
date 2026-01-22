import os
import stat
import sys
import unittest.mock
from test.support import unix_shell, requires_subprocess
from test.support import os_helper
from distutils.spawn import find_executable
from distutils.spawn import spawn
from distutils.errors import DistutilsExecError
from distutils.tests import support
@unittest.skipUnless(os.name in ('nt', 'posix'), 'Runs only under posix or nt')
def test_spawn(self):
    tmpdir = self.mkdtemp()
    if sys.platform != 'win32':
        exe = os.path.join(tmpdir, 'foo.sh')
        self.write_file(exe, '#!%s\nexit 1' % unix_shell)
    else:
        exe = os.path.join(tmpdir, 'foo.bat')
        self.write_file(exe, 'exit 1')
    os.chmod(exe, 511)
    self.assertRaises(DistutilsExecError, spawn, [exe])
    if sys.platform != 'win32':
        exe = os.path.join(tmpdir, 'foo.sh')
        self.write_file(exe, '#!%s\nexit 0' % unix_shell)
    else:
        exe = os.path.join(tmpdir, 'foo.bat')
        self.write_file(exe, 'exit 0')
    os.chmod(exe, 511)
    spawn([exe])