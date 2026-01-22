import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_rolling_apply_out_of_bounds(engine_and_raw):
    engine, raw = engine_and_raw
    vals = Series([1, 2, 3, 4])
    result = vals.rolling(10).apply(np.sum, engine=engine, raw=raw)
    assert result.isna().all()
    result = vals.rolling(10, min_periods=1).apply(np.sum, engine=engine, raw=raw)
    expected = Series([1, 3, 6, 10], dtype=float)
    tm.assert_almost_equal(result, expected)