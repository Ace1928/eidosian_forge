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
def test_resample_ohlc_result_odd_period(unit):
    rng = date_range('2013-12-30', '2014-01-07').as_unit(unit)
    index = rng.drop([Timestamp('2014-01-01'), Timestamp('2013-12-31'), Timestamp('2014-01-04'), Timestamp('2014-01-05')])
    df = DataFrame(data=np.arange(len(index)), index=index)
    result = df.resample('B').mean()
    expected = df.reindex(index=date_range(rng[0], rng[-1], freq='B').as_unit(unit))
    tm.assert_frame_equal(result, expected)