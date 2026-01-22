from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import period as libperiod
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.arm_slow
def test_getitem_seconds(self):
    didx = date_range(start='2013/01/01 09:00:00', freq='s', periods=4000)
    pidx = period_range(start='2013/01/01 09:00:00', freq='s', periods=4000)
    for idx in [didx, pidx]:
        values = ['2014', '2013/02', '2013/01/02', '2013/02/01 9h', '2013/02/01 09:00']
        for val in values:
            with pytest.raises(IndexError, match='only integers, slices'):
                idx[val]
        ser = Series(np.random.default_rng(2).random(len(idx)), index=idx)
        tm.assert_series_equal(ser['2013/01/01 10:00'], ser[3600:3660])
        tm.assert_series_equal(ser['2013/01/01 9h'], ser[:3600])
        for d in ['2013/01/01', '2013/01', '2013']:
            tm.assert_series_equal(ser[d], ser)