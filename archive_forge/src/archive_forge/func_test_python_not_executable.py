from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_python_not_executable(self):
    """Test completing a script that cannot be run directly."""
    prog = os.path.join(TEST_DIR, 'prog')
    with TempDir(prefix='test_dir_py', dir='.'):
        shutil.copy(prog, '.')
        self.sh.run_command('cd ' + os.getcwd())
        self.sh.run_command('chmod -x ./prog')
        self.assertIn('<<126>>', self.sh.run_command('./prog; echo "<<$?>>"'))
        self.assertEqual(self.sh.run_command('python ./prog basic f\t'), 'foo\r\n')