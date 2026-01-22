import itertools
import unittest
import numpy as np
from numba import jit, njit
from numba.core import types
from numba.tests import usecases
from numba.tests.support import TestCase
@TestCase.run_test_in_subprocess
def test_blackscholes_cnd(self):
    pyfunc = usecases.blackscholes_cnd
    cfunc = njit((types.float32,))(pyfunc)
    ds = (-0.5, 0, 0.5)
    for d in ds:
        args = (d,)
        self.assertEqual(pyfunc(*args), cfunc(*args), args)