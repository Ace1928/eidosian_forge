from datetime import (
import re
from dateutil.tz import (
import numpy as np
import pytest
import pytz
from pandas._libs import index as libindex
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_indexing_with_duplicate_datetimeindex(rand_series_with_duplicate_datetimeindex):
    ts = rand_series_with_duplicate_datetimeindex
    uniques = ts.index.unique()
    for date in uniques:
        result = ts[date]
        mask = ts.index == date
        total = (ts.index == date).sum()
        expected = ts[mask]
        if total > 1:
            tm.assert_series_equal(result, expected)
        else:
            tm.assert_almost_equal(result, expected.iloc[0])
        cp = ts.copy()
        cp[date] = 0
        expected = Series(np.where(mask, 0, ts), index=ts.index)
        tm.assert_series_equal(cp, expected)
    key = datetime(2000, 1, 6)
    with pytest.raises(KeyError, match=re.escape(repr(key))):
        ts[key]
    ts[datetime(2000, 1, 6)] = 0
    assert ts[datetime(2000, 1, 6)] == 0