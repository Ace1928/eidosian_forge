import warnings
import numpy as np
from numba import typed, njit, errors, typeof
from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method, overload_attribute
from numba.tests.support import (
def test_MyStructType(self):
    vs = np.arange(10, dtype=np.float64)
    ctr = 11
    first_expected_arr = vs.copy()
    first_got = ctor_by_class(vs, ctr)
    self.assertIsInstance(first_got, MyStruct)
    self.assertPreciseEqual(first_expected_arr, first_got.values)
    second_expected = first_expected_arr + ctr
    second_got = compute_fields(first_got)
    self.assertPreciseEqual(second_expected, second_got)
    self.assertEqual(first_got.counter, ctr)