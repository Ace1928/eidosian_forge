from __future__ import annotations
from datetime import datetime
import gc
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BaseMaskedArray
def test_index_groupby(self, simple_index):
    idx = simple_index[:5]
    to_groupby = np.array([1, 2, np.nan, 2, 1])
    tm.assert_dict_equal(idx.groupby(to_groupby), {1.0: idx[[0, 4]], 2.0: idx[[1, 3]]})
    to_groupby = DatetimeIndex([datetime(2011, 11, 1), datetime(2011, 12, 1), pd.NaT, datetime(2011, 12, 1), datetime(2011, 11, 1)], tz='UTC').values
    ex_keys = [Timestamp('2011-11-01'), Timestamp('2011-12-01')]
    expected = {ex_keys[0]: idx[[0, 4]], ex_keys[1]: idx[[1, 3]]}
    tm.assert_dict_equal(idx.groupby(to_groupby), expected)