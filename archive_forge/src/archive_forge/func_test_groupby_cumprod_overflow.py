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
def test_groupby_cumprod_overflow():
    df = DataFrame({'key': ['b'] * 4, 'value': 100000})
    actual = df.groupby('key')['value'].cumprod()
    expected = Series([100000, 10000000000, 1000000000000000, 7766279631452241920], name='value')
    tm.assert_series_equal(actual, expected)
    numpy_result = df.groupby('key', group_keys=False)['value'].apply(lambda x: x.cumprod())
    numpy_result.name = 'value'
    tm.assert_series_equal(actual, numpy_result)