from datetime import (
from itertools import product
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table
def test_pivot_table_sort_false(self):
    df = DataFrame({'a': ['d1', 'd4', 'd3'], 'col': ['a', 'b', 'c'], 'num': [23, 21, 34], 'year': ['2018', '2018', '2019']})
    result = df.pivot_table(index=['a', 'col'], columns='year', values='num', aggfunc='sum', sort=False)
    expected = DataFrame([[23, np.nan], [21, np.nan], [np.nan, 34]], columns=Index(['2018', '2019'], name='year'), index=MultiIndex.from_arrays([['d1', 'd4', 'd3'], ['a', 'b', 'c']], names=['a', 'col']))
    tm.assert_frame_equal(result, expected)