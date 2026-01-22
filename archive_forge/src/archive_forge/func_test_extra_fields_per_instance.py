import pytest
import pickle
from numpy.testing import assert_equal
from scipy._lib._bunch import _make_tuple_bunch
def test_extra_fields_per_instance(self):
    result1 = Result(x=1, y=2, z=3, w=-1, beta=0.0)
    result2 = Result(x=4, y=5, z=6, w=99, beta=1.0)
    assert_equal(result1.w, -1)
    assert_equal(result1.beta, 0.0)
    assert_equal(result1[:], (1, 2, 3))
    assert_equal(result2.w, 99)
    assert_equal(result2.beta, 1.0)
    assert_equal(result2[:], (4, 5, 6))