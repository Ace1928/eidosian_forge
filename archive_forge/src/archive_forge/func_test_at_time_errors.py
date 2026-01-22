from datetime import time
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import timezones
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('hour', ['1:00', '1:00AM', time(1), time(1, tzinfo=pytz.UTC)])
def test_at_time_errors(self, hour):
    dti = date_range('2018', periods=3, freq='h')
    df = DataFrame(list(range(len(dti))), index=dti)
    if getattr(hour, 'tzinfo', None) is None:
        result = df.at_time(hour)
        expected = df.iloc[1:2]
        tm.assert_frame_equal(result, expected)
    else:
        with pytest.raises(ValueError, match='Index must be timezone'):
            df.at_time(hour)