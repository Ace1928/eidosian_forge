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
def test_index_constructor_with_numpy_object_array_and_timestamp_tz_with_nan(self):
    result = Index(np.array([Timestamp('2019', tz='UTC'), np.nan], dtype=object))
    expected = DatetimeIndex([Timestamp('2019', tz='UTC'), pd.NaT])
    tm.assert_index_equal(result, expected)