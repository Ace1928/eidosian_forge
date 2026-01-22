import re
import pytest
from pandas._libs.tslibs import (
@pytest.mark.parametrize('kwargs,expected', [({'days': 1, 'seconds': 1}, offsets.Second(86401)), ({'days': -1, 'seconds': 1}, offsets.Second(-86399)), ({'hours': 1, 'minutes': 10}, offsets.Minute(70)), ({'hours': 1, 'minutes': -10}, offsets.Minute(50)), ({'weeks': 1}, offsets.Day(7)), ({'hours': 1}, offsets.Hour(1)), ({'hours': 1}, to_offset('60min')), ({'microseconds': 1}, offsets.Micro(1)), ({'microseconds': 0}, offsets.Nano(0))])
def test_to_offset_pd_timedelta(kwargs, expected):
    td = Timedelta(**kwargs)
    result = to_offset(td)
    assert result == expected