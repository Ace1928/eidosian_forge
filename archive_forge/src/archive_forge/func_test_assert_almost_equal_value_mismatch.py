import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_assert_almost_equal_value_mismatch():
    msg = 'expected 2\\.00000 but got 1\\.00000, with rtol=1e-05, atol=1e-08'
    with pytest.raises(AssertionError, match=msg):
        tm.assert_almost_equal(1, 2)