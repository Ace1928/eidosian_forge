from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
@pytest.mark.parametrize('dtype, unit', [['int64', 's'], ['int64', 'm'], ['int64', 'h'], ['timedelta64[s]', 's'], ['timedelta64[D]', 'D']])
def test_to_timedelta_units_dtypes(self, dtype, unit):
    arr = np.array([1] * 5, dtype=dtype)
    result = to_timedelta(arr, unit=unit)
    exp_dtype = 'm8[ns]' if dtype == 'int64' else 'm8[s]'
    expected = TimedeltaIndex([np.timedelta64(1, unit)] * 5, dtype=exp_dtype)
    tm.assert_index_equal(result, expected)