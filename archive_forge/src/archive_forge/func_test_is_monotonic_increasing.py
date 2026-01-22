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
@pytest.mark.parametrize('in_vals, out_vals', [([1, 2, 5, 3, 2, 0, 4, 5, -6, 1, 1], [True, False, False, True]), ([1, 2.1, np.inf, 3, 2, np.inf, -np.inf, 5, 11, 1, -np.inf], [True, False, True, False]), ([1, 2, np.nan, 3, 2, np.nan, np.nan, 5, -np.inf, 1, np.nan], [False, False, False, False])])
def test_is_monotonic_increasing(in_vals, out_vals):
    source_dict = {'A': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'], 'B': ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'd', 'd'], 'C': in_vals}
    df = DataFrame(source_dict)
    result = df.groupby('B').C.is_monotonic_increasing
    index = Index(list('abcd'), name='B')
    expected = Series(index=index, data=out_vals, name='C')
    tm.assert_series_equal(result, expected)
    expected = df.groupby(['B']).C.apply(lambda x: x.is_monotonic_increasing)
    tm.assert_series_equal(result, expected)