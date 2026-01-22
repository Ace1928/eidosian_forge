from __future__ import annotations
from functools import reduce
from itertools import product
import operator
import numpy as np
import pytest
from pandas.compat import PY312
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation import (
from pandas.core.computation.engines import ENGINES
from pandas.core.computation.expr import (
from pandas.core.computation.expressions import (
from pandas.core.computation.ops import (
from pandas.core.computation.scope import DEFAULT_GLOBALS
def test_set_inplace(using_copy_on_write, warn_copy_on_write):
    df = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    result_view = df[:]
    ser = df['A']
    with tm.assert_cow_warning(warn_copy_on_write):
        df.eval('A = B + C', inplace=True)
    expected = DataFrame({'A': [11, 13, 15], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    tm.assert_frame_equal(df, expected)
    if not using_copy_on_write:
        tm.assert_series_equal(ser, expected['A'])
        tm.assert_series_equal(result_view['A'], expected['A'])
    else:
        expected = Series([1, 2, 3], name='A')
        tm.assert_series_equal(ser, expected)
        tm.assert_series_equal(result_view['A'], expected)