import numpy as np
from numpy.testing import assert_equal, assert_raises
from statsmodels.tsa.arima.tools import (
def test_validate_basic():
    assert_equal(validate_basic([], 0, title='test'), [])
    assert_equal(validate_basic(0, 1), [0])
    assert_equal(validate_basic([0], 1), [0])
    assert_equal(validate_basic(np.array([1.2, 0.5 + 1j]), 2), np.array([1.2, 0.5 + 1j]))
    assert_equal(validate_basic([np.nan, -np.inf, np.inf], 3, allow_infnan=True), [np.nan, -np.inf, np.inf])
    assert_raises(ValueError, validate_basic, [], 1, title='test')
    assert_raises(ValueError, validate_basic, 0, 0)
    assert_raises(ValueError, validate_basic, 'a', 1)
    assert_raises(ValueError, validate_basic, None, 1)
    assert_raises(ValueError, validate_basic, np.nan, 1)
    assert_raises(ValueError, validate_basic, np.inf, 1)
    assert_raises(ValueError, validate_basic, -np.inf, 1)
    assert_raises(ValueError, validate_basic, [1, 2], 1)