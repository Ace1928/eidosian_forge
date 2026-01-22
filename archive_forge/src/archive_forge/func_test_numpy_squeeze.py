from copy import (
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
from pandas import (
import pandas._testing as tm
def test_numpy_squeeze(self):
    s = Series(range(2), dtype=np.float64)
    tm.assert_series_equal(np.squeeze(s), s)
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B')).reindex(columns=['A'])
    tm.assert_series_equal(np.squeeze(df), df['A'])