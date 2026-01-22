import sys
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types, errors, utils
from numba.tests.support import (captured_stdout, TestCase, EnableNRTStatsMixin)
def test_print_strings(self):
    pyfunc = print_string
    cfunc = njit((types.intp,))(pyfunc)
    with captured_stdout():
        cfunc(1)
        self.assertEqual(sys.stdout.getvalue(), '1 hop! 3.5\n')