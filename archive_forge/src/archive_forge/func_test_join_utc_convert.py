from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
def test_join_utc_convert(self, join_type):
    rng = date_range('1/1/2011', periods=100, freq='h', tz='utc')
    left = rng.tz_convert('US/Eastern')
    right = rng.tz_convert('Europe/Berlin')
    result = left.join(left[:-5], how=join_type)
    assert isinstance(result, DatetimeIndex)
    assert result.tz == left.tz
    result = left.join(right[:-5], how=join_type)
    assert isinstance(result, DatetimeIndex)
    assert result.tz is timezone.utc