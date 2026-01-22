from functools import partial
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
@pytest.mark.parametrize('sp_func, roll_func', [['kurtosis', 'kurt'], ['skew', 'skew']])
def test_time_rule_frame(raw, frame, sp_func, roll_func):
    sp_stats = pytest.importorskip('scipy.stats')
    compare_func = partial(getattr(sp_stats, sp_func), bias=False)
    win = 25
    frm = frame[::2].resample('B').mean()
    frame_result = getattr(frm.rolling(window=win, min_periods=10), roll_func)()
    last_date = frame_result.index[-1]
    prev_date = last_date - 24 * offsets.BDay()
    trunc_frame = frame[::2].truncate(prev_date, last_date)
    tm.assert_series_equal(frame_result.xs(last_date), trunc_frame.apply(compare_func, raw=raw), check_names=False)