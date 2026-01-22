import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def test_return_double_or_none(self):
    pyfunc = return_double_or_none
    cfunc = njit((types.boolean,))(pyfunc)
    for v in [True, False]:
        self.assertPreciseEqual(pyfunc(v), cfunc(v))