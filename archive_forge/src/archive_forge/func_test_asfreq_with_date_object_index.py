from datetime import datetime
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import MonthEnd
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_asfreq_with_date_object_index(self, frame_or_series):
    rng = date_range('1/1/2000', periods=20)
    ts = frame_or_series(np.random.default_rng(2).standard_normal(20), index=rng)
    ts2 = ts.copy()
    ts2.index = [x.date() for x in ts2.index]
    result = ts2.asfreq('4h', method='ffill')
    expected = ts.asfreq('4h', method='ffill')
    tm.assert_equal(result, expected)