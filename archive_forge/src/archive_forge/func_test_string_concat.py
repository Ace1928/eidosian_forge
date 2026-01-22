import itertools
import unittest
import numpy as np
from numba import jit, njit
from numba.core import types
from numba.tests import usecases
from numba.tests.support import TestCase
@TestCase.run_test_in_subprocess
def test_string_concat(self):
    pyfunc = usecases.string_concat
    cfunc = jit((types.int32, types.int32), forceobj=True)(pyfunc)
    xs = (-1, 0, 1)
    ys = (-1, 0, 1)
    for x, y in itertools.product(xs, ys):
        args = (x, y)
        self.assertEqual(pyfunc(*args), cfunc(*args), args)