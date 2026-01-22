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
def test_searchsorted_datetime(self):
    from .test_np_functions import searchsorted, searchsorted_left, searchsorted_right
    pyfunc_list = [searchsorted, searchsorted_left, searchsorted_right]
    cfunc_list = [jit(fn) for fn in pyfunc_list]

    def check(pyfunc, cfunc, a, v):
        expected = pyfunc(a, v)
        got = cfunc(a, v)
        self.assertPreciseEqual(expected, got)
    cases = self._get_testcases()
    for pyfunc, cfunc in zip(pyfunc_list, cfunc_list):
        for arr in cases:
            arr = np.sort(arr)
            for n in range(1, min(3, arr.size) + 1):
                idx = np.random.randint(0, arr.size, n)
                vs = arr[idx]
                if n == 1:
                    [v] = vs
                    check(pyfunc, cfunc, arr, v)
                check(pyfunc, cfunc, arr, vs)