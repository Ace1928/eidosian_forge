import sys
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types, errors, utils
from numba.tests.support import (captured_stdout, TestCase, EnableNRTStatsMixin)
def test_print_values(self):
    """
        Test printing a single argument value.
        """
    self.check_values(types.int32, (1, -234))
    self.check_values(types.int64, (1, -234, 123456789892843210, -123456789892843210))
    self.check_values(types.uint64, (1, 234, 123456789892843210, 2 ** 63 + 123))
    self.check_values(types.boolean, (True, False))
    self.check_values(types.float64, (1.5, 100.0 ** 10.0, float('nan')))
    self.check_values(types.complex64, (1 + 1j,))
    self.check_values(types.NPTimedelta('ms'), (np.timedelta64(100, 'ms'),))
    cfunc = njit((types.float32,))(print_value)
    with captured_stdout():
        cfunc(1.1)
        got = sys.stdout.getvalue()
        expect = '1.10000002384'
        self.assertTrue(got.startswith(expect))
        self.assertTrue(got.endswith('\n'))
    arraytype = types.Array(types.int32, 1, 'C')
    cfunc = njit((arraytype,))(print_value)
    with captured_stdout():
        cfunc(np.arange(10, dtype=np.int32))
        self.assertEqual(sys.stdout.getvalue(), '[0 1 2 3 4 5 6 7 8 9]\n')