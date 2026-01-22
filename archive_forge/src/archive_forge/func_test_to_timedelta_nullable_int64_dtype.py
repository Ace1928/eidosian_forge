from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
@pytest.mark.parametrize('expected_val, result_val', [[timedelta(days=2), 2], [None, None]])
def test_to_timedelta_nullable_int64_dtype(self, expected_val, result_val):
    expected = Series([timedelta(days=1), expected_val])
    result = to_timedelta(Series([1, result_val], dtype='Int64'), unit='days')
    tm.assert_series_equal(result, expected)