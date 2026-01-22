import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
def test_return_different_statement(self):
    pyfunc = return_different_statement
    cfunc = njit((types.boolean,))(pyfunc)
    for v in [True, False]:
        self.assertPreciseEqual(pyfunc(v), cfunc(v))