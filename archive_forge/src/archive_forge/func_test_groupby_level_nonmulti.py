from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_groupby_level_nonmulti():
    s = Series([1, 2, 3, 10, 4, 5, 20, 6], Index([1, 2, 3, 1, 4, 5, 2, 6], name='foo'))
    expected = Series([11, 22, 3, 4, 5, 6], Index(range(1, 7), name='foo'))
    result = s.groupby(level=0).sum()
    tm.assert_series_equal(result, expected)
    result = s.groupby(level=[0]).sum()
    tm.assert_series_equal(result, expected)
    result = s.groupby(level=-1).sum()
    tm.assert_series_equal(result, expected)
    result = s.groupby(level=[-1]).sum()
    tm.assert_series_equal(result, expected)
    msg = 'level > 0 or level < -1 only valid with MultiIndex'
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=1)
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=-2)
    msg = 'No group keys passed!'
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=[])
    msg = 'multiple levels only valid with MultiIndex'
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=[0, 0])
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=[0, 1])
    msg = 'level > 0 or level < -1 only valid with MultiIndex'
    with pytest.raises(ValueError, match=msg):
        s.groupby(level=[1])