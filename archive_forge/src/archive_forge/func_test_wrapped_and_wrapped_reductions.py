import pytest
import numpy as np
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import (
def test_wrapped_and_wrapped_reductions(self):
    a = self._get_array(2.0)
    float_equiv = a.astype(float)
    expected = np.hypot(float_equiv, float_equiv)
    res = np.hypot(a, a)
    assert res.dtype == a.dtype
    res_float = res.view(np.float64) * 2
    assert_array_equal(res_float, expected)
    res = np.hypot.reduce(a, keepdims=True)
    assert res.dtype == a.dtype
    expected = np.hypot.reduce(float_equiv, keepdims=True)
    assert res.view(np.float64) * 2 == expected