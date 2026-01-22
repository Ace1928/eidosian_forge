from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('window', [timedelta(days=3), Timedelta(days=3)])
def test_constructor_with_timedelta_window(window):
    n = 10
    df = DataFrame({'value': np.arange(n)}, index=date_range('2015-12-24', periods=n, freq='D'))
    expected_data = np.append([0.0, 1.0], np.arange(3.0, 27.0, 3))
    result = df.rolling(window=window).sum()
    expected = DataFrame({'value': expected_data}, index=date_range('2015-12-24', periods=n, freq='D'))
    tm.assert_frame_equal(result, expected)
    expected = df.rolling('3D').sum()
    tm.assert_frame_equal(result, expected)