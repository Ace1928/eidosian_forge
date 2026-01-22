from numba.tests.support import TestCase, linux_only
from numba.tests.gdb_support import needs_gdb, skip_unless_pexpect, GdbMIDriver
from unittest.mock import patch, Mock
from numba.core import datamodel
import numpy as np
from numba import typeof
import ctypes as ct
import unittest
def test_np_array_printer_simple_numeric_types_strided(self):
    n_tests = 30
    np.random.seed(0)
    for _ in range(n_tests):
        shape = np.random.randint(1, high=12, size=np.random.randint(1, 5))
        tmp = np.arange(np.prod(shape)).reshape(shape)
        slices = []
        for x in shape:
            start = np.random.randint(0, x)
            stop = np.random.randint(start + 1, max(start + 1, x + 3))
            step = np.random.randint(1, 3)
            strd = slice(start, stop, step)
            slices.append(strd)
        arr = tmp[tuple(slices)]
        self.check(arr)