from collections import deque
from datetime import (
from enum import Enum
import functools
import operator
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation import expressions as expr
from pandas.tests.frame.common import (
def test_frame_multiindex_operations_series_index_to_frame_index(self):
    df = DataFrame({2010: [1], 2020: [3]}, index=MultiIndex.from_product([['a'], ['b']], names=['scen', 'mod']))
    series = Series([10.0, 20.0, 30.0], index=MultiIndex.from_product([['a'], ['b'], [0, 1, 2]], names=['scen', 'mod', 'id']))
    expected = DataFrame({2010: [11.0, 21, 31.0], 2020: [13.0, 23.0, 33.0]}, index=MultiIndex.from_product([['a'], ['b'], [0, 1, 2]], names=['scen', 'mod', 'id']))
    result = df.add(series, axis=0)
    tm.assert_frame_equal(result, expected)