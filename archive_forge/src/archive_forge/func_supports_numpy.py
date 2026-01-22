import os
import re
import sys
import unittest
from numba.core import config
from numba.misc.gdb_hook import _confirm_gdb
from numba.misc.numba_gdbinfo import collect_gdbinfo
def supports_numpy(self):
    """Returns True if the underlying gdb implementation has NumPy support
           (and by extension Python support) False otherwise"""
    if not self.supports_python():
        return False
    cmd = 'python from __future__ import print_function;import numpy; print(numpy)'
    self.interpreter_exec('console', cmd)
    return "module 'numpy' from" in self._captured.before.decode()