import datetime as dt
from datetime import datetime
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz1', [None, 'UTC'])
@pytest.mark.parametrize('tz2', [None, 'UTC'])
def test_concat_NaT_series_dataframe_all_NaT(self, tz1, tz2):
    first = Series([pd.NaT, pd.NaT]).dt.tz_localize(tz1)
    second = DataFrame([[Timestamp('2015/01/01', tz=tz2)], [Timestamp('2016/01/01', tz=tz2)]], index=[2, 3])
    expected = DataFrame([pd.NaT, pd.NaT, Timestamp('2015/01/01', tz=tz2), Timestamp('2016/01/01', tz=tz2)])
    if tz1 != tz2:
        expected = expected.astype(object)
    result = concat([first, second])
    tm.assert_frame_equal(result, expected)