import sys
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types, errors, utils
from numba.tests.support import (captured_stdout, TestCase, EnableNRTStatsMixin)
def test_print_no_truncation(self):
    """ See: https://github.com/numba/numba/issues/3811
        """

    @jit(nopython=True)
    def foo():
        print(''.join(['a'] * 10000))
    with captured_stdout():
        foo()
        self.assertEqual(sys.stdout.getvalue(), ''.join(['a'] * 10000) + '\n')