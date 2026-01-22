import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('freq, periods, fill_method, limit', [('5B', 5, None, None), ('3B', 3, None, None), ('3B', 3, 'bfill', None), ('7B', 7, 'pad', 1), ('7B', 7, 'bfill', 3), ('14B', 14, None, None)])
def test_pct_change_periods_freq(self, datetime_frame, freq, periods, fill_method, limit):
    msg = "The 'fill_method' keyword being not None and the 'limit' keyword in DataFrame.pct_change are deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        rs_freq = datetime_frame.pct_change(freq=freq, fill_method=fill_method, limit=limit)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        rs_periods = datetime_frame.pct_change(periods, fill_method=fill_method, limit=limit)
    tm.assert_frame_equal(rs_freq, rs_periods)
    empty_ts = DataFrame(index=datetime_frame.index, columns=datetime_frame.columns)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        rs_freq = empty_ts.pct_change(freq=freq, fill_method=fill_method, limit=limit)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        rs_periods = empty_ts.pct_change(periods, fill_method=fill_method, limit=limit)
    tm.assert_frame_equal(rs_freq, rs_periods)