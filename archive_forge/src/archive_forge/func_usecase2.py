import sys
import unittest
import numpy as np
from numba import njit
from numba.core import types
from numba.tests.support import captured_stdout, TestCase
from numba.np import numpy_support
def usecase2(x, N):
    """
    Base on test1 of https://github.com/numba/numba/issues/381
    """
    for k in range(N):
        y = x[k]
        print(y.f1, y.s1, y.f2)