import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('arg', [[1, 2], ['a', 'b'], [Timestamp('2020-01-01', tz='Europe/London')] * 2])
def test_searchsorted_invalid_argument_dtype(self, arg):
    idx = TimedeltaIndex(['1 day', '2 days', '3 days'])
    msg = "value should be a 'Timedelta', 'NaT', or array of those. Got"
    with pytest.raises(TypeError, match=msg):
        idx.searchsorted(arg)