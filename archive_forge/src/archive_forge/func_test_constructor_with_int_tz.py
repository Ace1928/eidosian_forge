from __future__ import annotations
from datetime import (
from functools import partial
from operator import attrgetter
import dateutil
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
@pytest.mark.parametrize('klass', [Index, DatetimeIndex])
@pytest.mark.parametrize('box', [np.array, partial(np.array, dtype=object), list])
@pytest.mark.parametrize('tz, dtype', [('US/Pacific', 'datetime64[ns, US/Pacific]'), (None, 'datetime64[ns]')])
def test_constructor_with_int_tz(self, klass, box, tz, dtype):
    ts = Timestamp('2018-01-01', tz=tz).as_unit('ns')
    result = klass(box([ts._value]), dtype=dtype)
    expected = klass([ts])
    assert result == expected