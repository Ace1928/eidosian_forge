from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
def test_to_timedelta_invalid_errors_ignore(self):
    msg = "errors='ignore' is deprecated"
    invalid_data = 'apple'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = to_timedelta(invalid_data, errors='ignore')
    assert invalid_data == result
    invalid_data = ['apple', '1 days']
    expected = np.array(invalid_data, dtype=object)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = to_timedelta(invalid_data, errors='ignore')
    tm.assert_numpy_array_equal(expected, result)
    invalid_data = pd.Index(['apple', '1 days'])
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = to_timedelta(invalid_data, errors='ignore')
    tm.assert_index_equal(invalid_data, result)
    invalid_data = Series(['apple', '1 days'])
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = to_timedelta(invalid_data, errors='ignore')
    tm.assert_series_equal(invalid_data, result)