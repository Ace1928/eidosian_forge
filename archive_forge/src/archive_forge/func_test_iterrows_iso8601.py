import datetime
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
def test_iterrows_iso8601(self):
    s = DataFrame({'non_iso8601': ['M1701', 'M1802', 'M1903', 'M2004'], 'iso8601': date_range('2000-01-01', periods=4, freq='ME')})
    for k, v in s.iterrows():
        exp = s.loc[k]
        tm.assert_series_equal(v, exp)