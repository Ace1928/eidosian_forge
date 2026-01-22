import sys
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types, errors, utils
from numba.tests.support import (captured_stdout, TestCase, EnableNRTStatsMixin)
def test_print_w_kwarg_raises(self):

    @jit(nopython=True)
    def print_kwarg():
        print('x', flush=True)
    with self.assertRaises(errors.UnsupportedError) as raises:
        print_kwarg()
    expected = "Numba's print() function implementation does not support keyword arguments."
    self.assertIn(raises.exception.msg, expected)