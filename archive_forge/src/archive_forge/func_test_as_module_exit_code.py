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
def test_as_module_exit_code(self):
    cmdline = [sys.executable, '-m', 'numba']
    with self.assertRaises(AssertionError) as raises:
        run_cmd(cmdline)
    self.assertIn('process failed with code 1', str(raises.exception))