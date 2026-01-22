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
def test_explicit_tz_none(self):
    dti = date_range('2016-01-01', periods=10, tz='UTC')
    msg = "Passed data is timezone-aware, incompatible with 'tz=None'"
    with pytest.raises(ValueError, match=msg):
        DatetimeIndex(dti, tz=None)
    with pytest.raises(ValueError, match=msg):
        DatetimeIndex(np.array(dti), tz=None)
    msg = 'Cannot pass both a timezone-aware dtype and tz=None'
    with pytest.raises(ValueError, match=msg):
        DatetimeIndex([], dtype='M8[ns, UTC]', tz=None)