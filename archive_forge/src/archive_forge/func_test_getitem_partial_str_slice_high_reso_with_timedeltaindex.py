from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
from pandas.tseries.offsets import BDay
def test_getitem_partial_str_slice_high_reso_with_timedeltaindex(self):
    rng = timedelta_range('1 day 10:11:12', freq='us', periods=2000)
    ser = Series(np.arange(len(rng)), index=rng)
    result = ser['1 day 10:11:12':]
    expected = ser.iloc[0:]
    tm.assert_series_equal(result, expected)
    result = ser['1 day 10:11:12.001':]
    expected = ser.iloc[1000:]
    tm.assert_series_equal(result, expected)
    result = ser['1 days, 10:11:12.001001']
    assert result == ser.iloc[1001]