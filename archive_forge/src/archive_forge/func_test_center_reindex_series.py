from functools import partial
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
@pytest.mark.parametrize('roll_func', ['kurt', 'skew'])
def test_center_reindex_series(series, roll_func):
    s = [f'x{x:d}' for x in range(12)]
    series_xp = getattr(series.reindex(list(series.index) + s).rolling(window=25), roll_func)().shift(-12).reindex(series.index)
    series_rs = getattr(series.rolling(window=25, center=True), roll_func)()
    tm.assert_series_equal(series_xp, series_rs)