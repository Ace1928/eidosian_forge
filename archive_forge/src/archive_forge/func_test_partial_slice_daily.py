from datetime import datetime
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_partial_slice_daily(self):
    rng = date_range(freq='h', start=datetime(2005, 1, 31), periods=500)
    s = Series(np.arange(len(rng)), index=rng)
    result = s['2005-1-31']
    tm.assert_series_equal(result, s.iloc[:24])
    with pytest.raises(KeyError, match="^'2004-12-31 00'$"):
        s['2004-12-31 00']