from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import period as libperiod
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('idx_range', [date_range, period_range])
def test_getitem_day(self, idx_range):
    idx = idx_range(start='2013/01/01', freq='D', periods=400)
    values = ['2014', '2013/02', '2013/01/02', '2013/02/01 9h', '2013/02/01 09:00']
    for val in values:
        with pytest.raises(IndexError, match='only integers, slices'):
            idx[val]
    ser = Series(np.random.default_rng(2).random(len(idx)), index=idx)
    tm.assert_series_equal(ser['2013/01'], ser[0:31])
    tm.assert_series_equal(ser['2013/02'], ser[31:59])
    tm.assert_series_equal(ser['2014'], ser[365:])
    invalid = ['2013/02/01 9h', '2013/02/01 09:00']
    for val in invalid:
        with pytest.raises(KeyError, match=val):
            ser[val]