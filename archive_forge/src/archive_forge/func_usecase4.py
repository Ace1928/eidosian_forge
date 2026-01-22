import sys
import unittest
import numpy as np
from numba import njit
from numba.core import types
from numba.tests.support import captured_stdout, TestCase
from numba.np import numpy_support
def usecase4(x, N):
    """
    Base on test3 of https://github.com/numba/numba/issues/381
    """
    for k in range(N):
        y = x[k]
        print(y.f1, x.s1[k], y.f2)