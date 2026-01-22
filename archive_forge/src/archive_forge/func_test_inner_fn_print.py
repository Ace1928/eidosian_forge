import sys
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types, errors, utils
from numba.tests.support import (captured_stdout, TestCase, EnableNRTStatsMixin)
def test_inner_fn_print(self):

    @jit(nopython=True)
    def foo(x):
        print(x)

    @jit(nopython=True)
    def bar(x):
        foo(x)
        foo('hello')
    x = np.arange(5)
    with captured_stdout():
        bar(x)
        self.assertEqual(sys.stdout.getvalue(), '[0 1 2 3 4]\nhello\n')