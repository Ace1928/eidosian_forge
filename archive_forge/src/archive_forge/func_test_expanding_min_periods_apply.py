import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_expanding_min_periods_apply(engine_and_raw):
    engine, raw = engine_and_raw
    ser = Series(np.random.default_rng(2).standard_normal(50))
    result = ser.expanding(min_periods=30).apply(lambda x: x.mean(), raw=raw, engine=engine)
    assert result[:29].isna().all()
    tm.assert_almost_equal(result.iloc[-1], np.mean(ser[:50]))
    result = ser.expanding(min_periods=15).apply(lambda x: x.mean(), raw=raw, engine=engine)
    assert isna(result.iloc[13])
    assert notna(result.iloc[14])
    ser2 = Series(np.random.default_rng(2).standard_normal(20))
    result = ser2.expanding(min_periods=5).apply(lambda x: x.mean(), raw=raw, engine=engine)
    assert isna(result[3])
    assert notna(result[4])
    result0 = ser.expanding(min_periods=0).apply(lambda x: x.mean(), raw=raw, engine=engine)
    result1 = ser.expanding(min_periods=1).apply(lambda x: x.mean(), raw=raw, engine=engine)
    tm.assert_almost_equal(result0, result1)
    result = ser.expanding(min_periods=1).apply(lambda x: x.mean(), raw=raw, engine=engine)
    tm.assert_almost_equal(result.iloc[-1], np.mean(ser[:50]))