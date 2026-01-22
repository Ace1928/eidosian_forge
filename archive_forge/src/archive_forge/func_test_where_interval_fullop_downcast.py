from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_where_interval_fullop_downcast(self, frame_or_series):
    obj = frame_or_series([pd.Interval(0, 0)] * 2)
    other = frame_or_series([1.0, 2.0])
    msg = "Downcasting behavior in Series and DataFrame methods 'where'"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = obj.where(~obj.notna(), other)
    tm.assert_equal(res, other.astype(np.int64))
    with tm.assert_produces_warning(FutureWarning, match='Setting an item of incompatible dtype'):
        obj.mask(obj.notna(), other, inplace=True)
    tm.assert_equal(obj, other.astype(object))