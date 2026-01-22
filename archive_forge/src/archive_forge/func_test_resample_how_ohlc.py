from datetime import datetime
from functools import partial
import numpy as np
import pytest
import pytz
from pandas._libs import lib
from pandas._typing import DatetimeNaTType
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
from pandas.core.resample import (
from pandas.tseries import offsets
from pandas.tseries.offsets import Minute
@pytest.mark.parametrize('_index_start,_index_end,_index_name', [('1/1/2000 00:00:00', '1/1/2000 00:13:00', 'index')])
def test_resample_how_ohlc(series, unit):
    s = series
    s.index = s.index.as_unit(unit)
    grouplist = np.ones_like(s)
    grouplist[0] = 0
    grouplist[1:6] = 1
    grouplist[6:11] = 2
    grouplist[11:] = 3

    def _ohlc(group):
        if isna(group).all():
            return np.repeat(np.nan, 4)
        return [group.iloc[0], group.max(), group.min(), group.iloc[-1]]
    expected = DataFrame(s.groupby(grouplist).agg(_ohlc).values.tolist(), index=date_range('1/1/2000', periods=4, freq='5min', name='index').as_unit(unit), columns=['open', 'high', 'low', 'close'])
    result = s.resample('5min', closed='right', label='right').ohlc()
    tm.assert_frame_equal(result, expected)