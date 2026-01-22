import sys
import unittest
import numpy as np
from numba import njit
from numba.core import types
from numba.tests.support import captured_stdout, TestCase
from numba.np import numpy_support
def usecase1(arr1, arr2):
    """Base on https://github.com/numba/numba/issues/370

    Modified to add test-able side effect.
    """
    n1 = arr1.size
    n2 = arr2.size
    for i1 in range(n1):
        st1 = arr1[i1]
        for i2 in range(n2):
            st2 = arr2[i2]
            st2.row += st1.p * st2.p + st1.row - st1.col
        st1.p += st2.p
        st1.col -= st2.col