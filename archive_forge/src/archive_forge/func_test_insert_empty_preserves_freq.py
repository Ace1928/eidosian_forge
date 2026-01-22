from datetime import datetime
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
def test_insert_empty_preserves_freq(self, tz_naive_fixture):
    tz = tz_naive_fixture
    dti = DatetimeIndex([], tz=tz, freq='D')
    item = Timestamp('2017-04-05').tz_localize(tz)
    result = dti.insert(0, item)
    assert result.freq == dti.freq
    dti = DatetimeIndex([], tz=tz, freq='W-THU')
    result = dti.insert(0, item)
    assert result.freq is None