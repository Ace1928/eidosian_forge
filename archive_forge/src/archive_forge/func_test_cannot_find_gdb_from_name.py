import os
import subprocess
import sys
import threading
import json
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory
from unittest import mock
import unittest
from numba.tests.support import TestCase, linux_only
import numba.misc.numba_sysinfo as nsi
from numba.tests.gdb_support import needs_gdb
from numba.misc.numba_gdbinfo import collect_gdbinfo
from numba.misc.numba_gdbinfo import _GDBTestWrapper
def test_cannot_find_gdb_from_name(self):
    env = os.environ.copy()
    env['NUMBA_GDB_BINARY'] = 'THIS_IS_NOT_A_VALID_GDB_BINARY_NAME'
    cmdline = [sys.executable, '-m', 'numba', '-g']
    stdout, stderr = run_cmd(cmdline, env=env)
    self.assertIn('Testing gdb binary failed', stdout)
    self.assertIn('No such file or directory', stdout)
    self.assertIn("'THIS_IS_NOT_A_VALID_GDB_BINARY_NAME'", stdout)