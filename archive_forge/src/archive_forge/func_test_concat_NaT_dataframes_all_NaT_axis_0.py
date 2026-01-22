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
@pytest.mark.parametrize('item', [pd.NaT, Timestamp('20150101')])
def test_concat_NaT_dataframes_all_NaT_axis_0(self, tz1, tz2, item, using_array_manager):
    first = DataFrame([[pd.NaT], [pd.NaT]]).apply(lambda x: x.dt.tz_localize(tz1))
    second = DataFrame([item]).apply(lambda x: x.dt.tz_localize(tz2))
    result = concat([first, second], axis=0)
    expected = DataFrame(Series([pd.NaT, pd.NaT, item], index=[0, 1, 0]))
    expected = expected.apply(lambda x: x.dt.tz_localize(tz2))
    if tz1 != tz2:
        expected = expected.astype(object)
        if item is pd.NaT and (not using_array_manager):
            if tz1 is not None:
                expected.iloc[-1, 0] = np.nan
            else:
                expected.iloc[:-1, 0] = np.nan
    tm.assert_frame_equal(result, expected)