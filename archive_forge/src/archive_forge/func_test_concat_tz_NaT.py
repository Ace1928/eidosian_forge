import datetime as dt
from datetime import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('t1', ['2015-01-01', pytest.param(pd.NaT, marks=pytest.mark.xfail(reason='GH23037 incorrect dtype when concatenating'))])
def test_concat_tz_NaT(self, t1):
    ts1 = Timestamp(t1, tz='UTC')
    ts2 = Timestamp('2015-01-01', tz='UTC')
    ts3 = Timestamp('2015-01-01', tz='UTC')
    df1 = DataFrame([[ts1, ts2]])
    df2 = DataFrame([[ts3]])
    result = concat([df1, df2])
    expected = DataFrame([[ts1, ts2], [ts3, pd.NaT]], index=[0, 0])
    tm.assert_frame_equal(result, expected)