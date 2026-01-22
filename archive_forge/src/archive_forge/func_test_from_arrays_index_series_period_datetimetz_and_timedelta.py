from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize(('idx1', 'idx2'), [(pd.period_range('2011-01-01', freq='D', periods=3), pd.period_range('2015-01-01', freq='h', periods=3)), (date_range('2015-01-01 10:00', freq='D', periods=3, tz='US/Eastern'), date_range('2015-01-01 10:00', freq='h', periods=3, tz='Asia/Tokyo')), (pd.timedelta_range('1 days', freq='D', periods=3), pd.timedelta_range('2 hours', freq='h', periods=3))])
def test_from_arrays_index_series_period_datetimetz_and_timedelta(idx1, idx2):
    result = MultiIndex.from_arrays([idx1, idx2])
    tm.assert_index_equal(result.get_level_values(0), idx1)
    tm.assert_index_equal(result.get_level_values(1), idx2)
    result2 = MultiIndex.from_arrays([Series(idx1), Series(idx2)])
    tm.assert_index_equal(result2.get_level_values(0), idx1)
    tm.assert_index_equal(result2.get_level_values(1), idx2)
    tm.assert_index_equal(result, result2)