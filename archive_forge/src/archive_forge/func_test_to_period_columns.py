import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_to_period_columns(self):
    dr = date_range('1/1/2000', '1/1/2001')
    df = DataFrame(np.random.default_rng(2).standard_normal((len(dr), 5)), index=dr)
    df['mix'] = 'a'
    df = df.T
    pts = df.to_period(axis=1)
    exp = df.copy()
    exp.columns = period_range('1/1/2000', '1/1/2001')
    tm.assert_frame_equal(pts, exp)
    pts = df.to_period('M', axis=1)
    tm.assert_index_equal(pts.columns, exp.columns.asfreq('M'))