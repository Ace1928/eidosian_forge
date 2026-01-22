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
@pytest.mark.parametrize('tzstr', ['US/Eastern', 'dateutil/US/Eastern'])
def test_dti_convert_datetime_list(self, tzstr):
    dr = date_range('2012-06-02', periods=10, tz=tzstr, name='foo')
    dr2 = DatetimeIndex(list(dr), name='foo', freq='D')
    tm.assert_index_equal(dr, dr2)