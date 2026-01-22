from datetime import datetime
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import MonthEnd
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_asfreq_normalize(self, frame_or_series):
    rng = date_range('1/1/2000 09:30', periods=20)
    norm = date_range('1/1/2000', periods=20)
    vals = np.random.default_rng(2).standard_normal((20, 3))
    obj = DataFrame(vals, index=rng)
    expected = DataFrame(vals, index=norm)
    if frame_or_series is Series:
        obj = obj[0]
        expected = expected[0]
    result = obj.asfreq('D', normalize=True)
    tm.assert_equal(result, expected)