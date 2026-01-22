from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_aggregating_timedelta_and_datetime():
    df = DataFrame({'clientid': ['A', 'B', 'C'], 'datetime': [np.datetime64('2017-02-01 00:00:00')] * 3})
    df['time_delta_zero'] = df.datetime - df.datetime
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        result = df.groupby('clientid').apply(lambda ddf: Series({'clientid_age': ddf.time_delta_zero.min(), 'date': ddf.datetime.min()}))
    expected = DataFrame({'clientid': ['A', 'B', 'C'], 'clientid_age': [np.timedelta64(0, 'D')] * 3, 'date': [np.datetime64('2017-02-01 00:00:00')] * 3}).set_index('clientid')
    tm.assert_frame_equal(result, expected)