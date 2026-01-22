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
@needs_gdb
def test_gdb_status_from_module(self):
    cmdline = [sys.executable, '-m', 'numba', '-g']
    o, _ = run_cmd(cmdline)
    self.assertIn('GDB info', o)
    self.assertIn('Numba printing extension support', o)