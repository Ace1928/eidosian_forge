import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import FloatingArray
@pytest.mark.parametrize('zero, negative', [(0, False), (0.0, False), (-0.0, True)])
def test_divide_by_zero(zero, negative):
    a = pd.array([0, 1, -1, None], dtype='Int64')
    result = a / zero
    expected = FloatingArray(np.array([np.nan, np.inf, -np.inf, 1], dtype='float64'), np.array([False, False, False, True]))
    if negative:
        expected *= -1
    tm.assert_extension_array_equal(result, expected)