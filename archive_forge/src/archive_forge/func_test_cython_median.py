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
def test_cython_median():
    arr = np.random.randn(1000)
    arr[::2] = np.nan
    df = DataFrame(arr)
    labels = np.random.randint(0, 50, size=1000).astype(float)
    labels[::17] = np.nan
    result = df.groupby(labels).median()
    exp = df.groupby(labels).agg(nanops.nanmedian)
    tm.assert_frame_equal(result, exp)
    df = DataFrame(np.random.randn(1000, 5))
    rs = df.groupby(labels).agg(np.median)
    xp = df.groupby(labels).median()
    tm.assert_frame_equal(rs, xp)