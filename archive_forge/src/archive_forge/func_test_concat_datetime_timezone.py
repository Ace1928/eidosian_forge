import datetime as dt
from datetime import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_datetime_timezone(self):
    idx1 = date_range('2011-01-01', periods=3, freq='h', tz='Europe/Paris')
    idx2 = date_range(start=idx1[0], end=idx1[-1], freq='h')
    df1 = DataFrame({'a': [1, 2, 3]}, index=idx1)
    df2 = DataFrame({'b': [1, 2, 3]}, index=idx2)
    result = concat([df1, df2], axis=1)
    exp_idx = DatetimeIndex(['2011-01-01 00:00:00+01:00', '2011-01-01 01:00:00+01:00', '2011-01-01 02:00:00+01:00'], dtype='M8[ns, Europe/Paris]', freq='h')
    expected = DataFrame([[1, 1], [2, 2], [3, 3]], index=exp_idx, columns=['a', 'b'])
    tm.assert_frame_equal(result, expected)
    idx3 = date_range('2011-01-01', periods=3, freq='h', tz='Asia/Tokyo')
    df3 = DataFrame({'b': [1, 2, 3]}, index=idx3)
    result = concat([df1, df3], axis=1)
    exp_idx = DatetimeIndex(['2010-12-31 15:00:00+00:00', '2010-12-31 16:00:00+00:00', '2010-12-31 17:00:00+00:00', '2010-12-31 23:00:00+00:00', '2011-01-01 00:00:00+00:00', '2011-01-01 01:00:00+00:00']).as_unit('ns')
    expected = DataFrame([[np.nan, 1], [np.nan, 2], [np.nan, 3], [1, np.nan], [2, np.nan], [3, np.nan]], index=exp_idx, columns=['a', 'b'])
    tm.assert_frame_equal(result, expected)
    result = concat([df1.resample('h').mean(), df2.resample('h').mean()], sort=True)
    expected = DataFrame({'a': [1, 2, 3] + [np.nan] * 3, 'b': [np.nan] * 3 + [1, 2, 3]}, index=idx1.append(idx1))
    tm.assert_frame_equal(result, expected)