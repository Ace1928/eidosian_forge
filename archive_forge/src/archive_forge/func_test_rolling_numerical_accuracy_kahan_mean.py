from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('add', [0.0, 2.0])
def test_rolling_numerical_accuracy_kahan_mean(add, unit):
    dti = DatetimeIndex([Timestamp('19700101 09:00:00'), Timestamp('19700101 09:00:03'), Timestamp('19700101 09:00:06')]).as_unit(unit)
    df = DataFrame({'A': [3002399751580331.0 + add, -0.0, -0.0]}, index=dti)
    result = df.resample('1s').ffill().rolling('3s', closed='left', min_periods=3).mean()
    dates = date_range('19700101 09:00:00', periods=7, freq='s', unit=unit)
    expected = DataFrame({'A': [np.nan, np.nan, np.nan, 3002399751580330.5, 2001599834386887.2, 1000799917193443.6, 0.0]}, index=dates)
    tm.assert_frame_equal(result, expected)