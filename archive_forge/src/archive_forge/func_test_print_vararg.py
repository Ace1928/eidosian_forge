import sys
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types, errors, utils
from numba.tests.support import (captured_stdout, TestCase, EnableNRTStatsMixin)
def test_print_vararg(self):
    pyfunc = print_vararg
    cfunc = jit(nopython=True)(pyfunc)
    with captured_stdout():
        cfunc(1, (2, 3), (4, 5j))
        self.assertEqual(sys.stdout.getvalue(), '1 (2, 3) 4 5j\n')
    pyfunc = print_string_vararg
    cfunc = jit(nopython=True)(pyfunc)
    with captured_stdout():
        cfunc(1, (2, 3), (4, 5j))
        self.assertEqual(sys.stdout.getvalue(), '1 hop! (2, 3) 4 5j\n')