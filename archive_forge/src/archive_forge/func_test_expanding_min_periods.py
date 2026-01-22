import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('func,static_comp', [('sum', np.sum), ('mean', np.mean), ('max', np.max), ('min', np.min)], ids=['sum', 'mean', 'max', 'min'])
def test_expanding_min_periods(func, static_comp):
    ser = Series(np.random.default_rng(2).standard_normal(50))
    msg = "The 'axis' keyword in Series.expanding is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = getattr(ser.expanding(min_periods=30, axis=0), func)()
    assert result[:29].isna().all()
    tm.assert_almost_equal(result.iloc[-1], static_comp(ser[:50]))
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = getattr(ser.expanding(min_periods=15, axis=0), func)()
    assert isna(result.iloc[13])
    assert notna(result.iloc[14])
    ser2 = Series(np.random.default_rng(2).standard_normal(20))
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = getattr(ser2.expanding(min_periods=5, axis=0), func)()
    assert isna(result[3])
    assert notna(result[4])
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result0 = getattr(ser.expanding(min_periods=0, axis=0), func)()
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result1 = getattr(ser.expanding(min_periods=1, axis=0), func)()
    tm.assert_almost_equal(result0, result1)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = getattr(ser.expanding(min_periods=1, axis=0), func)()
    tm.assert_almost_equal(result.iloc[-1], static_comp(ser[:50]))