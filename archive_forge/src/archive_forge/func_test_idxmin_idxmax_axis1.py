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
def test_idxmin_idxmax_axis1():
    df = DataFrame(np.random.randn(10, 4), columns=['A', 'B', 'C', 'D'])
    df['A'] = [1, 2, 3, 1, 2, 3, 1, 2, 3, 4]
    gb = df.groupby('A')
    res = gb.idxmax(axis=1)
    alt = df.iloc[:, 1:].idxmax(axis=1)
    indexer = res.index.get_level_values(1)
    tm.assert_series_equal(alt[indexer], res.droplevel('A'))
    df['E'] = date_range('2016-01-01', periods=10)
    gb2 = df.groupby('A')
    msg = "reduction operation 'argmax' not allowed for this dtype"
    with pytest.raises(TypeError, match=msg):
        gb2.idxmax(axis=1)