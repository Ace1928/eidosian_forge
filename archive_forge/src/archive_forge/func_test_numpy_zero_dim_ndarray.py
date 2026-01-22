import operator
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import FloatingArray
@pytest.mark.parametrize('other', [0, 0.5])
def test_numpy_zero_dim_ndarray(other):
    arr = pd.array([1, None, 2])
    result = arr + np.array(other)
    expected = arr + other
    tm.assert_equal(result, expected)