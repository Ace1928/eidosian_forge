from numba.tests.support import TestCase, linux_only
from numba.tests.gdb_support import needs_gdb, skip_unless_pexpect, GdbMIDriver
from unittest.mock import patch, Mock
from numba.core import datamodel
import numpy as np
from numba import typeof
import ctypes as ct
import unittest
def test_np_array_printer_nested_array_structured_dtype(self):
    n = 4
    m = 3
    dt = np.dtype([('x', np.int16, (2,)), ('y', np.float64)], align=True)
    arr = np.zeros(m * n, dtype=dt).reshape(m, n)
    rep = self.get_gdb_repr(arr)
    self.assertIn('array[Exception:', rep)
    self.assertIn('Unsupported sub-type', rep)
    self.assertIn('nestedarray(int16', rep)