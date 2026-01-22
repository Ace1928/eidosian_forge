from datetime import (
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_date_range_localize(self, unit):
    rng = date_range('3/11/2012 03:00', periods=15, freq='h', tz='US/Eastern', unit=unit)
    rng2 = DatetimeIndex(['3/11/2012 03:00', '3/11/2012 04:00'], dtype=f'M8[{unit}, US/Eastern]')
    rng3 = date_range('3/11/2012 03:00', periods=15, freq='h', unit=unit)
    rng3 = rng3.tz_localize('US/Eastern')
    tm.assert_index_equal(rng._with_freq(None), rng3)
    val = rng[0]
    exp = Timestamp('3/11/2012 03:00', tz='US/Eastern')
    assert val.hour == 3
    assert exp.hour == 3
    assert val == exp
    tm.assert_index_equal(rng[:2], rng2)