import numpy as np
import pytest
from pandas import MultiIndex
import pandas._testing as tm
def test_isin_missing(nulls_fixture):
    mi1 = MultiIndex.from_tuples([(1, nulls_fixture)])
    mi2 = MultiIndex.from_tuples([(1, 1), (1, 2)])
    result = mi2.isin(mi1)
    expected = np.array([False, False])
    tm.assert_numpy_array_equal(result, expected)