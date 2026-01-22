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
def test_datetimeindex_constructor_misc(self):
    arr = ['1/1/2005', '1/2/2005', 'Jn 3, 2005', '2005-01-04']
    msg = "(\\(')?Unknown datetime string format(:', 'Jn 3, 2005'\\))?"
    with pytest.raises(ValueError, match=msg):
        DatetimeIndex(arr)
    arr = ['1/1/2005', '1/2/2005', '1/3/2005', '2005-01-04']
    idx1 = DatetimeIndex(arr)
    arr = [datetime(2005, 1, 1), '1/2/2005', '1/3/2005', '2005-01-04']
    idx2 = DatetimeIndex(arr)
    arr = [Timestamp(datetime(2005, 1, 1)), '1/2/2005', '1/3/2005', '2005-01-04']
    idx3 = DatetimeIndex(arr)
    arr = np.array(['1/1/2005', '1/2/2005', '1/3/2005', '2005-01-04'], dtype='O')
    idx4 = DatetimeIndex(arr)
    idx5 = DatetimeIndex(['12/05/2007', '25/01/2008'], dayfirst=True)
    idx6 = DatetimeIndex(['2007/05/12', '2008/01/25'], dayfirst=False, yearfirst=True)
    tm.assert_index_equal(idx5, idx6)
    for other in [idx2, idx3, idx4]:
        assert (idx1.values == other.values).all()