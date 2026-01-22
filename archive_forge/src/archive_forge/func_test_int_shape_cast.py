import numpy as np
import unittest
from numba import jit, vectorize, int8, int16, int32
from numba.tests.support import TestCase
from numba.tests.enum_usecases import (Color, Shape, Shake,
def test_int_shape_cast(self):

    def pyfun_empty(x):
        return np.empty((x, x), dtype='int64').fill(-1)

    def pyfun_zeros(x):
        return np.zeros((x, x), dtype='int64')

    def pyfun_ones(x):
        return np.ones((x, x), dtype='int64')
    for pyfun in [pyfun_empty, pyfun_zeros, pyfun_ones]:
        cfunc = jit(nopython=True)(pyfun)
        for member in IntEnumWithNegatives:
            if member >= 0:
                self.assertPreciseEqual(pyfun(member), cfunc(member))