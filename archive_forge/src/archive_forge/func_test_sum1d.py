import itertools
import unittest
import numpy as np
from numba import jit, njit
from numba.core import types
from numba.tests import usecases
from numba.tests.support import TestCase
@TestCase.run_test_in_subprocess
def test_sum1d(self):
    pyfunc = usecases.sum1d
    cfunc = njit((types.int32, types.int32))(pyfunc)
    ss = (-1, 0, 1, 100, 200)
    es = (-1, 0, 1, 100, 200)
    for args in itertools.product(ss, es):
        self.assertEqual(pyfunc(*args), cfunc(*args), args)