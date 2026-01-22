from datetime import (
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
@pytest.mark.parametrize('tz', [None, 'UTC', 'Europe/Berlin', pytz.FixedOffset(-60)])
def test_intersection_dst_transition(self, tz):
    idx1 = date_range('2020-03-27', periods=5, freq='D', tz=tz)
    idx2 = date_range('2020-03-30', periods=5, freq='D', tz=tz)
    result = idx1.intersection(idx2)
    expected = date_range('2020-03-30', periods=2, freq='D', tz=tz)
    tm.assert_index_equal(result, expected)
    index1 = date_range('2021-10-28', periods=3, freq='D', tz='Europe/London')
    index2 = date_range('2021-10-30', periods=4, freq='D', tz='Europe/London')
    result = index1.union(index2)
    expected = date_range('2021-10-28', periods=6, freq='D', tz='Europe/London')
    tm.assert_index_equal(result, expected)