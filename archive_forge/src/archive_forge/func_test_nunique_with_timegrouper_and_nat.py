from datetime import (
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouper
from pandas.core.groupby.ops import BinGrouper
def test_nunique_with_timegrouper_and_nat(self):
    test = DataFrame({'time': [Timestamp('2016-06-28 09:35:35'), pd.NaT, Timestamp('2016-06-28 16:46:28')], 'data': ['1', '2', '3']})
    grouper = Grouper(key='time', freq='h')
    result = test.groupby(grouper)['data'].nunique()
    expected = test[test.time.notnull()].groupby(grouper)['data'].nunique()
    expected.index = expected.index._with_freq(None)
    tm.assert_series_equal(result, expected)