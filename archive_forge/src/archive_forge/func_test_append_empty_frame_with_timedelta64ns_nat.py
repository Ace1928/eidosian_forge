import datetime as dt
from itertools import combinations
import dateutil
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype_str', ['datetime64[ns, UTC]', 'datetime64[ns]', 'Int64', 'int64'])
@pytest.mark.parametrize('val', [1, 'NaT'])
def test_append_empty_frame_with_timedelta64ns_nat(self, dtype_str, val, using_array_manager):
    df = DataFrame(columns=['a']).astype(dtype_str)
    other = DataFrame({'a': [np.timedelta64(val, 'ns')]})
    result = df._append(other, ignore_index=True)
    expected = other.astype(object)
    if isinstance(val, str) and dtype_str != 'int64' and (not using_array_manager):
        expected = DataFrame({'a': [np.nan]}, dtype=object)
    tm.assert_frame_equal(result, expected)