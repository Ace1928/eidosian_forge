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
def test_groupby_multiindex_series_keys_len_equal_group_axis():
    index_array = [['x', 'x'], ['a', 'b'], ['k', 'k']]
    index_names = ['first', 'second', 'third']
    ri = MultiIndex.from_arrays(index_array, names=index_names)
    s = Series(data=[1, 2], index=ri)
    result = s.groupby(['first', 'third']).sum()
    index_array = [['x'], ['k']]
    index_names = ['first', 'third']
    ei = MultiIndex.from_arrays(index_array, names=index_names)
    expected = Series([3], index=ei)
    tm.assert_series_equal(result, expected)