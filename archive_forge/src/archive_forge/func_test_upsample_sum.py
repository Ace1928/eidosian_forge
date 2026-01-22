from datetime import datetime
from operator import methodcaller
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
@pytest.mark.parametrize('method, method_args, expected_values', [('sum', {}, [1, 0, 1]), ('sum', {'min_count': 0}, [1, 0, 1]), ('sum', {'min_count': 1}, [1, np.nan, 1]), ('sum', {'min_count': 2}, [np.nan, np.nan, np.nan]), ('prod', {}, [1, 1, 1]), ('prod', {'min_count': 0}, [1, 1, 1]), ('prod', {'min_count': 1}, [1, np.nan, 1]), ('prod', {'min_count': 2}, [np.nan, np.nan, np.nan])])
def test_upsample_sum(method, method_args, expected_values):
    ser = Series(1, index=date_range('2017', periods=2, freq='h'))
    resampled = ser.resample('30min')
    index = pd.DatetimeIndex(['2017-01-01T00:00:00', '2017-01-01T00:30:00', '2017-01-01T01:00:00'], dtype='M8[ns]', freq='30min')
    result = methodcaller(method, **method_args)(resampled)
    expected = Series(expected_values, index=index)
    tm.assert_series_equal(result, expected)