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
def test_describe(self, df, gb, gni):
    expected_index = Index([1, 3], name='A')
    expected_col = MultiIndex(levels=[['B'], ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']], codes=[[0] * 8, list(range(8))])
    expected = DataFrame([[1.0, 2.0, np.nan, 2.0, 2.0, 2.0, 2.0, 2.0], [0.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]], index=expected_index, columns=expected_col)
    result = gb.describe()
    tm.assert_frame_equal(result, expected)
    expected = expected.reset_index()
    result = gni.describe()
    tm.assert_frame_equal(result, expected)