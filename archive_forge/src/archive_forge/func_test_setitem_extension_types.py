from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.base import _registry as ea_registry
from pandas.core.dtypes.common import is_object_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tseries.offsets import BDay
@pytest.mark.parametrize('obj,dtype', [(Period('2020-01'), PeriodDtype('M')), (Interval(left=0, right=5), IntervalDtype('int64', 'right')), (Timestamp('2011-01-01', tz='US/Eastern'), DatetimeTZDtype(unit='s', tz='US/Eastern'))])
def test_setitem_extension_types(self, obj, dtype):
    expected = DataFrame({'idx': [1, 2, 3], 'obj': Series([obj] * 3, dtype=dtype)})
    df = DataFrame({'idx': [1, 2, 3]})
    df['obj'] = obj
    tm.assert_frame_equal(df, expected)