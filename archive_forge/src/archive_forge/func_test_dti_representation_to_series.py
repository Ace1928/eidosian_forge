from datetime import datetime
import dateutil.tz
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_dti_representation_to_series(self, unit):
    idx1 = DatetimeIndex([], freq='D')
    idx2 = DatetimeIndex(['2011-01-01'], freq='D')
    idx3 = DatetimeIndex(['2011-01-01', '2011-01-02'], freq='D')
    idx4 = DatetimeIndex(['2011-01-01', '2011-01-02', '2011-01-03'], freq='D')
    idx5 = DatetimeIndex(['2011-01-01 09:00', '2011-01-01 10:00', '2011-01-01 11:00'], freq='h', tz='Asia/Tokyo')
    idx6 = DatetimeIndex(['2011-01-01 09:00', '2011-01-01 10:00', NaT], tz='US/Eastern')
    idx7 = DatetimeIndex(['2011-01-01 09:00', '2011-01-02 10:15'])
    exp1 = 'Series([], dtype: datetime64[ns])'
    exp2 = '0   2011-01-01\ndtype: datetime64[ns]'
    exp3 = '0   2011-01-01\n1   2011-01-02\ndtype: datetime64[ns]'
    exp4 = '0   2011-01-01\n1   2011-01-02\n2   2011-01-03\ndtype: datetime64[ns]'
    exp5 = '0   2011-01-01 09:00:00+09:00\n1   2011-01-01 10:00:00+09:00\n2   2011-01-01 11:00:00+09:00\ndtype: datetime64[ns, Asia/Tokyo]'
    exp6 = '0   2011-01-01 09:00:00-05:00\n1   2011-01-01 10:00:00-05:00\n2                         NaT\ndtype: datetime64[ns, US/Eastern]'
    exp7 = '0   2011-01-01 09:00:00\n1   2011-01-02 10:15:00\ndtype: datetime64[ns]'
    with pd.option_context('display.width', 300):
        for idx, expected in zip([idx1, idx2, idx3, idx4, idx5, idx6, idx7], [exp1, exp2, exp3, exp4, exp5, exp6, exp7]):
            ser = Series(idx.as_unit(unit))
            result = repr(ser)
            assert result == expected.replace('[ns', f'[{unit}')