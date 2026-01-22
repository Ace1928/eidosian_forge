from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_ops_not_as_index(reduction_func):
    if reduction_func in ('corrwith', 'nth', 'ngroup'):
        pytest.skip(f'GH 5755: Test not applicable for {reduction_func}')
    df = DataFrame(np.random.default_rng(2).integers(0, 5, size=(100, 2)), columns=['a', 'b'])
    expected = getattr(df.groupby('a'), reduction_func)()
    if reduction_func == 'size':
        expected = expected.rename('size')
    expected = expected.reset_index()
    if reduction_func != 'size':
        expected['a'] = expected['a'].astype(df['a'].dtype)
    g = df.groupby('a', as_index=False)
    result = getattr(g, reduction_func)()
    tm.assert_frame_equal(result, expected)
    result = g.agg(reduction_func)
    tm.assert_frame_equal(result, expected)
    result = getattr(g['b'], reduction_func)()
    tm.assert_frame_equal(result, expected)
    result = g['b'].agg(reduction_func)
    tm.assert_frame_equal(result, expected)