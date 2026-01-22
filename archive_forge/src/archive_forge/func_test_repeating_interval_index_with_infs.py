import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
@pytest.mark.xfail(not IS64, reason='GH 23440')
@pytest.mark.parametrize('intervals', [[Interval(-np.inf, 0.0), Interval(0.0, 1.0)], [Interval(-np.inf, -2.0), Interval(-2.0, -1.0)], [Interval(-1.0, 0.0), Interval(0.0, np.inf)], [Interval(1.0, 2.0), Interval(2.0, np.inf)]])
def test_repeating_interval_index_with_infs(intervals):
    interval_index = Index(intervals * 51)
    expected = np.arange(1, 102, 2, dtype=np.intp)
    result = interval_index.get_indexer_for([intervals[1]])
    tm.assert_equal(result, expected)