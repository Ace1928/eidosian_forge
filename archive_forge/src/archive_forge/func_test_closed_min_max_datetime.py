from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('input_dtype', ['int', 'float'])
@pytest.mark.parametrize('func,closed,expected', [('min', 'right', [0.0, 0, 0, 1, 2, 3, 4, 5, 6, 7]), ('min', 'both', [0.0, 0, 0, 0, 1, 2, 3, 4, 5, 6]), ('min', 'neither', [np.nan, 0, 0, 1, 2, 3, 4, 5, 6, 7]), ('min', 'left', [np.nan, 0, 0, 0, 1, 2, 3, 4, 5, 6]), ('max', 'right', [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), ('max', 'both', [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), ('max', 'neither', [np.nan, 0, 1, 2, 3, 4, 5, 6, 7, 8]), ('max', 'left', [np.nan, 0, 1, 2, 3, 4, 5, 6, 7, 8])])
def test_closed_min_max_datetime(input_dtype, func, closed, expected):
    ser = Series(data=np.arange(10).astype(input_dtype), index=date_range('2000', periods=10))
    result = getattr(ser.rolling('3D', closed=closed), func)()
    expected = Series(expected, index=ser.index)
    tm.assert_series_equal(result, expected)