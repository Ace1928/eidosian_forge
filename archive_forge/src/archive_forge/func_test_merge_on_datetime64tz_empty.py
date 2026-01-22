from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
def test_merge_on_datetime64tz_empty(self):
    dtz = pd.DatetimeTZDtype(tz='UTC')
    right = DataFrame({'date': DatetimeIndex(['2018'], dtype=dtz), 'value': [4.0], 'date2': DatetimeIndex(['2019'], dtype=dtz)}, columns=['date', 'value', 'date2'])
    left = right[:0]
    result = left.merge(right, on='date')
    expected = DataFrame({'date': Series(dtype=dtz), 'value_x': Series(dtype=float), 'date2_x': Series(dtype=dtz), 'value_y': Series(dtype=float), 'date2_y': Series(dtype=dtz)}, columns=['date', 'value_x', 'date2_x', 'value_y', 'date2_y'])
    tm.assert_frame_equal(result, expected)