from datetime import datetime
from operator import methodcaller
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
@pytest.mark.xfail(reason="if TimeGrouper is used included, 'nth' doesn't work yet")
def test_aggregate_nth():
    """Check TimeGrouper's aggregation is identical as normal groupby."""
    data = np.random.default_rng(2).standard_normal((20, 4))
    normal_df = DataFrame(data, columns=['A', 'B', 'C', 'D'])
    normal_df['key'] = [1, 2, 3, 4, 5] * 4
    dt_df = DataFrame(data, columns=['A', 'B', 'C', 'D'])
    dt_df['key'] = [datetime(2013, 1, 1), datetime(2013, 1, 2), datetime(2013, 1, 3), datetime(2013, 1, 4), datetime(2013, 1, 5)] * 4
    normal_grouped = normal_df.groupby('key')
    dt_grouped = dt_df.groupby(Grouper(key='key', freq='D'))
    expected = normal_grouped.nth(3)
    expected.index = date_range(start='2013-01-01', freq='D', periods=5, name='key')
    dt_result = dt_grouped.nth(3)
    tm.assert_frame_equal(expected, dt_result)