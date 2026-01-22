from datetime import (
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
def test_groupby_with_timezone_selection(self):
    df = DataFrame({'factor': np.random.default_rng(2).integers(0, 3, size=60), 'time': date_range('01/01/2000 00:00', periods=60, freq='s', tz='UTC')})
    df1 = df.groupby('factor').max()['time']
    df2 = df.groupby('factor')['time'].max()
    tm.assert_series_equal(df1, df2)