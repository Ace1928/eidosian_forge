from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', [None, 'UTC', 'US/Eastern'])
def test_insert_invalid_na(self, tz):
    idx = DatetimeIndex(['2017-01-01'], tz=tz)
    item = np.timedelta64('NaT')
    result = idx.insert(0, item)
    expected = Index([item] + list(idx), dtype=object)
    tm.assert_index_equal(result, expected)