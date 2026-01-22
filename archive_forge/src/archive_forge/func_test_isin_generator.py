import numpy as np
import pytest
from pandas import MultiIndex
import pandas._testing as tm
def test_isin_generator():
    midx = MultiIndex.from_tuples([(1, 2)])
    result = midx.isin((x for x in [(1, 2)]))
    expected = np.array([True])
    tm.assert_numpy_array_equal(result, expected)