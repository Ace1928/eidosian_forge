from __future__ import annotations
from datetime import timedelta
import operator
import numpy as np
import pytest
from pandas._libs.tslibs import tz_compare
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
def test_setitem_different_tz_raises(self):
    data = np.array([1, 2, 3], dtype='M8[ns]')
    arr = DatetimeArray._from_sequence(data, copy=False, dtype=DatetimeTZDtype(tz='US/Central'))
    with pytest.raises(TypeError, match='Cannot compare tz-naive and tz-aware'):
        arr[0] = pd.Timestamp('2000')
    ts = pd.Timestamp('2000', tz='US/Eastern')
    arr[0] = ts
    assert arr[0] == ts.tz_convert('US/Central')