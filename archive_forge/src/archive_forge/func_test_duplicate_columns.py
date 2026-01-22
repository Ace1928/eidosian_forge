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
def test_duplicate_columns(request, groupby_func, as_index):
    if groupby_func == 'corrwith':
        msg = 'GH#50845 - corrwith fails when there are duplicate columns'
        request.node.add_marker(pytest.mark.xfail(reason=msg))
    df = DataFrame([[1, 3, 6], [1, 4, 7], [2, 5, 8]], columns=list('abb'))
    args = get_groupby_method_args(groupby_func, df)
    gb = df.groupby('a', as_index=as_index)
    result = getattr(gb, groupby_func)(*args)
    expected_df = df.set_axis(['a', 'b', 'c'], axis=1)
    expected_args = get_groupby_method_args(groupby_func, expected_df)
    expected_gb = expected_df.groupby('a', as_index=as_index)
    expected = getattr(expected_gb, groupby_func)(*expected_args)
    if groupby_func not in ('size', 'ngroup', 'cumcount'):
        expected = expected.rename(columns={'c': 'b'})
    tm.assert_equal(result, expected)