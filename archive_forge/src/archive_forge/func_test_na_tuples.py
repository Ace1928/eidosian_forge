from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_unsigned_integer_dtype
from pandas.core.dtypes.dtypes import IntervalDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
import pandas.core.common as com
def test_na_tuples(self):
    na_tuple = [(0, 1), (np.nan, np.nan), (2, 3)]
    idx_na_tuple = IntervalIndex.from_tuples(na_tuple)
    idx_na_element = IntervalIndex.from_tuples([(0, 1), np.nan, (2, 3)])
    tm.assert_index_equal(idx_na_tuple, idx_na_element)