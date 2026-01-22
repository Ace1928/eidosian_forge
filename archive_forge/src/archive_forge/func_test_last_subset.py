import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_last_subset(self, frame_or_series):
    ts = DataFrame(np.random.default_rng(2).standard_normal((100, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=100, freq='12h'))
    ts = tm.get_obj(ts, frame_or_series)
    with tm.assert_produces_warning(FutureWarning, match=last_deprecated_msg):
        result = ts.last('10d')
    assert len(result) == 20
    ts = DataFrame(np.random.default_rng(2).standard_normal((30, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=30, freq='D'))
    ts = tm.get_obj(ts, frame_or_series)
    with tm.assert_produces_warning(FutureWarning, match=last_deprecated_msg):
        result = ts.last('10d')
    assert len(result) == 10
    with tm.assert_produces_warning(FutureWarning, match=last_deprecated_msg):
        result = ts.last('21D')
    expected = ts['2000-01-10':]
    tm.assert_equal(result, expected)
    with tm.assert_produces_warning(FutureWarning, match=last_deprecated_msg):
        result = ts.last('21D')
    expected = ts[-21:]
    tm.assert_equal(result, expected)
    with tm.assert_produces_warning(FutureWarning, match=last_deprecated_msg):
        result = ts[:0].last('3ME')
    tm.assert_equal(result, ts[:0])