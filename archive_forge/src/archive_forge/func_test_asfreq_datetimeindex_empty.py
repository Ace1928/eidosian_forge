from datetime import datetime
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import MonthEnd
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_asfreq_datetimeindex_empty(self, frame_or_series):
    index = DatetimeIndex(['2016-09-29 11:00'])
    expected = frame_or_series(index=index, dtype=object).asfreq('h')
    result = frame_or_series([3], index=index.copy()).asfreq('h')
    tm.assert_index_equal(expected.index, result.index)