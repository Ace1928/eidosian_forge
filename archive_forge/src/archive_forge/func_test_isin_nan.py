import numpy as np
import pytest
from pandas import MultiIndex
import pandas._testing as tm
def test_isin_nan():
    idx = MultiIndex.from_arrays([['foo', 'bar'], [1.0, np.nan]])
    tm.assert_numpy_array_equal(idx.isin([('bar', np.nan)]), np.array([False, True]))
    tm.assert_numpy_array_equal(idx.isin([('bar', float('nan'))]), np.array([False, True]))