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
def test_mean_on_timedelta():
    df = DataFrame({'time': pd.to_timedelta(range(10)), 'cat': ['A', 'B'] * 5})
    result = df.groupby('cat')['time'].mean()
    expected = Series(pd.to_timedelta([4, 5]), name='time', index=Index(['A', 'B'], name='cat'))
    tm.assert_series_equal(result, expected)