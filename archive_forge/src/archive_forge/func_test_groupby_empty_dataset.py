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
@pytest.mark.parametrize('dtype', [int, float, object])
@pytest.mark.parametrize('kwargs', [{'percentiles': [0.1, 0.2, 0.3], 'include': 'all', 'exclude': None}, {'percentiles': [0.1, 0.2, 0.3], 'include': None, 'exclude': ['int']}, {'percentiles': [0.1, 0.2, 0.3], 'include': ['int'], 'exclude': None}])
def test_groupby_empty_dataset(dtype, kwargs):
    df = DataFrame([[1, 2, 3]], columns=['A', 'B', 'C'], dtype=dtype)
    df['B'] = df['B'].astype(int)
    df['C'] = df['C'].astype(float)
    result = df.iloc[:0].groupby('A').describe(**kwargs)
    expected = df.groupby('A').describe(**kwargs).reset_index(drop=True).iloc[:0]
    tm.assert_frame_equal(result, expected)
    result = df.iloc[:0].groupby('A').B.describe(**kwargs)
    expected = df.groupby('A').B.describe(**kwargs).reset_index(drop=True).iloc[:0]
    expected.index = Index([])
    tm.assert_frame_equal(result, expected)