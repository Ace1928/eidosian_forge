import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
def test_intersect1d_array_like(self):

    class Test:

        def __array__(self):
            return np.arange(3)
    a = Test()
    res = intersect1d(a, a)
    assert_array_equal(res, a)
    res = intersect1d([1, 2, 3], [1, 2, 3])
    assert_array_equal(res, [1, 2, 3])