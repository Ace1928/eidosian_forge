import itertools
import math
import sys
import unittest
import warnings
import numpy as np
from numba import njit, types
from numba.tests.support import TestCase
from numba.np import numpy_support
def test_ldexp(self):
    pyfunc = ldexp
    cfunc = njit(pyfunc)
    for fltty in (types.float32, types.float64):
        for args in [(2.5, -2), (2.5, 1), (0.0, 0), (0.0, 1), (-0.0, 0), (-0.0, 1), (float('inf'), 0), (float('-inf'), 0), (float('nan'), 0)]:
            msg = 'for input %r' % (args,)
            self.assertPreciseEqual(cfunc(*args), pyfunc(*args))