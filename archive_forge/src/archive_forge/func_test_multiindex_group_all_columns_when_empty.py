import builtins
from io import StringIO
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.tests.groupby import get_groupby_method_args
from pandas.util import _test_decorators as td
def test_multiindex_group_all_columns_when_empty(groupby_func):
    df = DataFrame({'a': [], 'b': [], 'c': []}).set_index(['a', 'b', 'c'])
    gb = df.groupby(['a', 'b', 'c'], group_keys=False)
    method = getattr(gb, groupby_func)
    args = get_groupby_method_args(groupby_func, df)
    result = method(*args).index
    expected = df.index
    tm.assert_index_equal(result, expected)