from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_categorical_series_repr_datetime_ordered(self):
    idx = date_range('2011-01-01 09:00', freq='H', periods=5)
    s = Series(Categorical(idx, ordered=True))
    exp = '0   2011-01-01 09:00:00\n1   2011-01-01 10:00:00\n2   2011-01-01 11:00:00\n3   2011-01-01 12:00:00\n4   2011-01-01 13:00:00\ndtype: category\nCategories (5, datetime64[ns]): [2011-01-01 09:00:00 < 2011-01-01 10:00:00 < 2011-01-01 11:00:00 <\n                                 2011-01-01 12:00:00 < 2011-01-01 13:00:00]'
    assert repr(s) == exp
    idx = date_range('2011-01-01 09:00', freq='H', periods=5, tz='US/Eastern')
    s = Series(Categorical(idx, ordered=True))
    exp = '0   2011-01-01 09:00:00-05:00\n1   2011-01-01 10:00:00-05:00\n2   2011-01-01 11:00:00-05:00\n3   2011-01-01 12:00:00-05:00\n4   2011-01-01 13:00:00-05:00\ndtype: category\nCategories (5, datetime64[ns, US/Eastern]): [2011-01-01 09:00:00-05:00 < 2011-01-01 10:00:00-05:00 <\n                                             2011-01-01 11:00:00-05:00 < 2011-01-01 12:00:00-05:00 <\n                                             2011-01-01 13:00:00-05:00]'
    assert repr(s) == exp