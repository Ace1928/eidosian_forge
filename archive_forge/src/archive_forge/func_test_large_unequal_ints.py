import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['Int64', 'int64'])
def test_large_unequal_ints(dtype):
    left = Series([1577840521123000], dtype=dtype)
    right = Series([1577840521123543], dtype=dtype)
    with pytest.raises(AssertionError, match='Series are different'):
        tm.assert_series_equal(left, right)