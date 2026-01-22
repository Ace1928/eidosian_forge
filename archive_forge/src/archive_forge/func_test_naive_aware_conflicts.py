from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
def test_naive_aware_conflicts(self):
    start, end = (datetime(2009, 1, 1), datetime(2010, 1, 1))
    naive = date_range(start, end, freq=BDay(), tz=None)
    aware = date_range(start, end, freq=BDay(), tz='Asia/Hong_Kong')
    msg = 'tz-naive.*tz-aware'
    with pytest.raises(TypeError, match=msg):
        naive.join(aware)
    with pytest.raises(TypeError, match=msg):
        aware.join(naive)