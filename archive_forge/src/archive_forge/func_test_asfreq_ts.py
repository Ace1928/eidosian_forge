from datetime import datetime
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import MonthEnd
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_asfreq_ts(self, frame_or_series):
    index = period_range(freq='Y', start='1/1/2001', end='12/31/2010')
    obj = DataFrame(np.random.default_rng(2).standard_normal((len(index), 3)), index=index)
    obj = tm.get_obj(obj, frame_or_series)
    result = obj.asfreq('D', how='end')
    exp_index = index.asfreq('D', how='end')
    assert len(result) == len(obj)
    tm.assert_index_equal(result.index, exp_index)
    result = obj.asfreq('D', how='start')
    exp_index = index.asfreq('D', how='start')
    assert len(result) == len(obj)
    tm.assert_index_equal(result.index, exp_index)