from datetime import time
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import timezones
from pandas import (
import pandas._testing as tm
def test_at_time_raises(self, frame_or_series):
    obj = DataFrame([[1, 2, 3], [4, 5, 6]])
    obj = tm.get_obj(obj, frame_or_series)
    msg = 'Index must be DatetimeIndex'
    with pytest.raises(TypeError, match=msg):
        obj.at_time('00:00')