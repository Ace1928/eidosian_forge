import contextlib
import itertools
import re
import unittest
import warnings
import numpy as np
from numba import jit, vectorize, njit
from numba.np.numpy_support import numpy_version
from numba.core import types, config
from numba.core.errors import TypingError
from numba.tests.support import TestCase, tag, skip_parfors_unsupported
from numba.np import npdatetime_helpers, numpy_support
def test_div(self):
    """
        Test the division of a timedelta by numeric types
        """

    def arr_div(a, b):
        return a / b
    py_func = arr_div
    cfunc = njit(arr_div)
    test_cases = [(np.ones(3, TIMEDELTA_M), np.ones(3, TIMEDELTA_M)), (np.ones(3, TIMEDELTA_M), np.ones(3, TIMEDELTA_Y)), (np.ones(3, TIMEDELTA_Y), np.ones(3, TIMEDELTA_M)), (np.ones(3, TIMEDELTA_Y), np.ones(3, TIMEDELTA_Y)), (np.ones(3, TIMEDELTA_M), 1), (np.ones(3, TIMEDELTA_M), np.ones(3, np.int64)), (np.ones(3, TIMEDELTA_M), np.ones(3, np.float64))]
    for a, b in test_cases:
        self.assertTrue(np.array_equal(py_func(a, b), cfunc(a, b)))