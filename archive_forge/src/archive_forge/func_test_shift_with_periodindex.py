import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_shift_with_periodindex(self, frame_or_series):
    ps = DataFrame(np.arange(4, dtype=float), index=pd.period_range('2020-01-01', periods=4))
    ps = tm.get_obj(ps, frame_or_series)
    shifted = ps.shift(1)
    unshifted = shifted.shift(-1)
    tm.assert_index_equal(shifted.index, ps.index)
    tm.assert_index_equal(unshifted.index, ps.index)
    if frame_or_series is DataFrame:
        tm.assert_numpy_array_equal(unshifted.iloc[:, 0].dropna().values, ps.iloc[:-1, 0].values)
    else:
        tm.assert_numpy_array_equal(unshifted.dropna().values, ps.values[:-1])
    shifted2 = ps.shift(1, 'D')
    shifted3 = ps.shift(1, offsets.Day())
    tm.assert_equal(shifted2, shifted3)
    tm.assert_equal(ps, shifted2.shift(-1, 'D'))
    msg = 'does not match PeriodIndex freq'
    with pytest.raises(ValueError, match=msg):
        ps.shift(freq='W')
    shifted4 = ps.shift(1, freq='D')
    tm.assert_equal(shifted2, shifted4)
    shifted5 = ps.shift(1, freq=offsets.Day())
    tm.assert_equal(shifted5, shifted4)