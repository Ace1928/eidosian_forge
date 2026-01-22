from numba.tests.support import TestCase, linux_only
from numba.tests.gdb_support import needs_gdb, skip_unless_pexpect, GdbMIDriver
from unittest.mock import patch, Mock
from numba.core import datamodel
import numpy as np
from numba import typeof
import ctypes as ct
import unittest
def test_pretty_print(self):
    if not self._gdb_has_numpy():
        _msg = 'Cannot find gdb with NumPy support'
        self.skipTest(_msg)
    self._subprocess_test_runner('test_pretty_print')