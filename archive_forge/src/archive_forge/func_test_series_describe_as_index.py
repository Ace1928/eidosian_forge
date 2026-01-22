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
@pytest.mark.parametrize('keys', ['key1', ['key1', 'key2']])
def test_series_describe_as_index(as_index, keys):
    df = DataFrame({'key1': ['one', 'two', 'two', 'three', 'two'], 'key2': ['one', 'two', 'two', 'three', 'two'], 'foo2': [1, 2, 4, 4, 6]})
    gb = df.groupby(keys, as_index=as_index)['foo2']
    result = gb.describe()
    expected = DataFrame({'key1': ['one', 'three', 'two'], 'count': [1.0, 1.0, 3.0], 'mean': [1.0, 4.0, 4.0], 'std': [np.nan, np.nan, 2.0], 'min': [1.0, 4.0, 2.0], '25%': [1.0, 4.0, 3.0], '50%': [1.0, 4.0, 4.0], '75%': [1.0, 4.0, 5.0], 'max': [1.0, 4.0, 6.0]})
    if len(keys) == 2:
        expected.insert(1, 'key2', expected['key1'])
    if as_index:
        expected = expected.set_index(keys)
    tm.assert_frame_equal(result, expected)