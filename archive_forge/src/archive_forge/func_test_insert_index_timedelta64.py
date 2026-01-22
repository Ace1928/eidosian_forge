from __future__ import annotations
from datetime import (
import itertools
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas as pd
import pandas._testing as tm
def test_insert_index_timedelta64(self):
    obj = pd.TimedeltaIndex(['1 day', '2 day', '3 day', '4 day'])
    assert obj.dtype == 'timedelta64[ns]'
    exp = pd.TimedeltaIndex(['1 day', '10 day', '2 day', '3 day', '4 day'])
    self._assert_insert_conversion(obj, pd.Timedelta('10 day'), exp, 'timedelta64[ns]')
    for item in [pd.Timestamp('2012-01-01'), 1]:
        result = obj.insert(1, item)
        expected = obj.astype(object).insert(1, item)
        assert expected.dtype == object
        tm.assert_index_equal(result, expected)