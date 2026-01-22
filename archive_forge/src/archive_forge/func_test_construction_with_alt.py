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
@pytest.mark.parametrize('kwargs', [{'tz': 'dtype.tz'}, {'dtype': 'dtype'}, {'dtype': 'dtype', 'tz': 'dtype.tz'}])
def test_construction_with_alt(self, kwargs, tz_aware_fixture):
    tz = tz_aware_fixture
    i = date_range('20130101', periods=5, freq='h', tz=tz)
    kwargs = {key: attrgetter(val)(i) for key, val in kwargs.items()}
    result = DatetimeIndex(i, **kwargs)
    tm.assert_index_equal(i, result)