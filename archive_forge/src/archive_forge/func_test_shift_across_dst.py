from hypothesis import (
import pytest
import pytz
import pandas as pd
from pandas._testing._hypothesis import (
@given(YQM_OFFSET)
def test_shift_across_dst(offset):
    assume(not offset.normalize)
    dti = pd.date_range(start='2017-10-30 12:00:00', end='2017-11-06', freq='D', tz='US/Eastern')
    assert (dti.hour == 12).all()
    res = dti + offset
    assert (res.hour == 12).all()