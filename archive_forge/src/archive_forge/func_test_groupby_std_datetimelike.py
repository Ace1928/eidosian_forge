from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_groupby_std_datetimelike(warn_copy_on_write):
    tdi = pd.timedelta_range('1 Day', periods=10000)
    ser = Series(tdi)
    ser[::5] *= 2
    df = ser.to_frame('A').copy()
    df['B'] = ser + Timestamp(0)
    df['C'] = ser + Timestamp(0, tz='UTC')
    df.iloc[-1] = pd.NaT
    gb = df.groupby(list(range(5)) * 2000)
    result = gb.std()
    td1 = Timedelta('2887 days 11:21:02.326710176')
    td4 = Timedelta('2886 days 00:42:34.664668096')
    exp_ser = Series([td1 * 2, td1, td1, td1, td4], index=np.arange(5))
    expected = DataFrame({'A': exp_ser, 'B': exp_ser, 'C': exp_ser})
    tm.assert_frame_equal(result, expected)