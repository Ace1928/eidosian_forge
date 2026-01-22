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
@pytest.mark.parametrize('indexer', ['a', ['a'], pytest.param([True, False], marks=pytest.mark.xfail(reason='Boolean indexer incorrectly setting inplace', strict=False))])
@pytest.mark.parametrize('value, set_value', [(1, 5), (1.0, 5.0), (Timestamp('2020-12-31'), Timestamp('2021-12-31')), ('a', 'b')])
def test_setitem_not_operating_inplace(self, value, set_value, indexer):
    df = DataFrame({'a': value}, index=[0, 1])
    expected = df.copy()
    view = df[:]
    df[indexer] = set_value
    tm.assert_frame_equal(view, expected)