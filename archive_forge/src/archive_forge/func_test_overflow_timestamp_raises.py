from datetime import (
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas._testing as tm
def test_overflow_timestamp_raises(self):
    msg = 'Result is too large'
    a = Timestamp('2101-01-01 00:00:00').as_unit('ns')
    b = Timestamp('1688-01-01 00:00:00').as_unit('ns')
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        a - b
    assert a - b.to_pydatetime() == a.to_pydatetime() - b