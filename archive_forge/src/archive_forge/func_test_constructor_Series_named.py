import array
from collections import (
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
import functools
import re
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.arrays import (
def test_constructor_Series_named(self):
    a = Series([1, 2, 3], index=['a', 'b', 'c'], name='x')
    df = DataFrame(a)
    assert df.columns[0] == 'x'
    tm.assert_index_equal(df.index, a.index)
    arr = np.random.default_rng(2).standard_normal(10)
    s = Series(arr, name='x')
    df = DataFrame(s)
    expected = DataFrame({'x': s})
    tm.assert_frame_equal(df, expected)
    s = Series(arr, index=range(3, 13))
    df = DataFrame(s)
    expected = DataFrame({0: s})
    tm.assert_frame_equal(df, expected)
    msg = 'Shape of passed values is \\(10, 1\\), indices imply \\(10, 2\\)'
    with pytest.raises(ValueError, match=msg):
        DataFrame(s, columns=[1, 2])
    a = Series([], name='x', dtype=object)
    df = DataFrame(a)
    assert df.columns[0] == 'x'
    s1 = Series(arr, name='x')
    df = DataFrame([s1, arr]).T
    expected = DataFrame({'x': s1, 'Unnamed 0': arr}, columns=['x', 'Unnamed 0'])
    tm.assert_frame_equal(df, expected)
    df = DataFrame([arr, s1]).T
    expected = DataFrame({1: s1, 0: arr}, columns=[0, 1])
    tm.assert_frame_equal(df, expected)