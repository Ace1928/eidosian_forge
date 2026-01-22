import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_groupby_agg_extension_timedelta_cumsum_with_named_aggregation():
    expected = DataFrame({'td': {0: pd.Timedelta('0 days 01:00:00'), 1: pd.Timedelta('0 days 01:15:00'), 2: pd.Timedelta('0 days 01:15:00')}})
    df = DataFrame({'td': Series(['0 days 01:00:00', '0 days 00:15:00', '0 days 01:15:00'], dtype='timedelta64[ns]'), 'grps': ['a', 'a', 'b']})
    gb = df.groupby('grps')
    result = gb.agg(td=('td', 'cumsum'))
    tm.assert_frame_equal(result, expected)