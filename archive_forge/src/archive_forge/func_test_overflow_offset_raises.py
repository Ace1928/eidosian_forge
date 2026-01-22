from datetime import (
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas._testing as tm
def test_overflow_offset_raises(self):
    stamp = Timestamp('2017-01-13 00:00:00').as_unit('ns')
    offset_overflow = 20169940 * offsets.Day(1)
    lmsg2 = "Cannot cast -?20169940 days \\+?00:00:00 to unit='ns' without overflow"
    with pytest.raises(OutOfBoundsTimedelta, match=lmsg2):
        stamp + offset_overflow
    with pytest.raises(OutOfBoundsTimedelta, match=lmsg2):
        offset_overflow + stamp
    with pytest.raises(OutOfBoundsTimedelta, match=lmsg2):
        stamp - offset_overflow
    stamp = Timestamp('2000/1/1').as_unit('ns')
    offset_overflow = to_offset('D') * 100 ** 5
    lmsg3 = "Cannot cast -?10000000000 days \\+?00:00:00 to unit='ns' without overflow"
    with pytest.raises(OutOfBoundsTimedelta, match=lmsg3):
        stamp + offset_overflow
    with pytest.raises(OutOfBoundsTimedelta, match=lmsg3):
        offset_overflow + stamp
    with pytest.raises(OutOfBoundsTimedelta, match=lmsg3):
        stamp - offset_overflow