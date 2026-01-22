import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat import IS64
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.xfail(not IS64, reason='GH 23440')
@pytest.mark.parametrize('base', [101, 1010])
def test_reindex_behavior_with_interval_index(self, base):
    ser = Series(range(base), index=IntervalIndex.from_arrays(range(base), range(1, base + 1)))
    expected_result = Series([np.nan, 0], index=[np.nan, 1.0], dtype=float)
    result = ser.reindex(index=[np.nan, 1.0])
    tm.assert_series_equal(result, expected_result)