from datetime import (
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
def test_groupby_max_datetime64(self):
    df = DataFrame({'A': Timestamp('20130101'), 'B': np.arange(5)})
    expected = df.groupby('A')['A'].apply(lambda x: x.max()).astype('M8[s]')
    result = df.groupby('A')['A'].max()
    tm.assert_series_equal(result, expected)