import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_sample_does_not_modify_weights(self):
    result = np.array([np.nan, 1, np.nan])
    expected = result.copy()
    ser = Series([1, 2, 3])
    ser.sample(weights=result)
    tm.assert_numpy_array_equal(result, expected)
    df = DataFrame({'values': [1, 1, 1], 'weights': [1, np.nan, np.nan]})
    expected = df['weights'].copy()
    df.sample(frac=1.0, replace=True, weights='weights')
    result = df['weights']
    tm.assert_series_equal(result, expected)