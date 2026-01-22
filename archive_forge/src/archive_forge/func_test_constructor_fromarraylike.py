import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_constructor_fromarraylike(self):
    idx = period_range('2007-01', periods=20, freq='M')
    tm.assert_index_equal(PeriodIndex(idx.values), idx)
    tm.assert_index_equal(PeriodIndex(list(idx.values)), idx)
    msg = 'freq not specified and cannot be inferred'
    with pytest.raises(ValueError, match=msg):
        PeriodIndex(idx.asi8)
    with pytest.raises(ValueError, match=msg):
        PeriodIndex(list(idx.asi8))
    msg = "'Period' object is not iterable"
    with pytest.raises(TypeError, match=msg):
        PeriodIndex(data=Period('2007', freq='Y'))
    result = PeriodIndex(iter(idx))
    tm.assert_index_equal(result, idx)
    result = PeriodIndex(idx)
    tm.assert_index_equal(result, idx)
    result = PeriodIndex(idx, freq='M')
    tm.assert_index_equal(result, idx)
    result = PeriodIndex(idx, freq=offsets.MonthEnd())
    tm.assert_index_equal(result, idx)
    assert result.freq == 'ME'
    result = PeriodIndex(idx, freq='2M')
    tm.assert_index_equal(result, idx.asfreq('2M'))
    assert result.freq == '2ME'
    result = PeriodIndex(idx, freq=offsets.MonthEnd(2))
    tm.assert_index_equal(result, idx.asfreq('2M'))
    assert result.freq == '2ME'
    result = PeriodIndex(idx, freq='D')
    exp = idx.asfreq('D', 'e')
    tm.assert_index_equal(result, exp)