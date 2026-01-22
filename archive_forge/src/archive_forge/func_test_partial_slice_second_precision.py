from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_partial_slice_second_precision(self):
    rng = date_range(start=datetime(2005, 1, 1, 0, 0, 59, microsecond=999990), periods=20, freq='us')
    s = Series(np.arange(20), rng)
    tm.assert_series_equal(s['2005-1-1 00:00'], s.iloc[:10])
    tm.assert_series_equal(s['2005-1-1 00:00:59'], s.iloc[:10])
    tm.assert_series_equal(s['2005-1-1 00:01'], s.iloc[10:])
    tm.assert_series_equal(s['2005-1-1 00:01:00'], s.iloc[10:])
    assert s[Timestamp('2005-1-1 00:00:59.999990')] == s.iloc[0]
    with pytest.raises(KeyError, match='2005-1-1 00:00:00'):
        s['2005-1-1 00:00:00']