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
def test_constructor_error_msgs(self):
    msg = 'Empty data passed with indices specified.'
    with pytest.raises(ValueError, match=msg):
        DataFrame(np.empty(0), index=[1])
    msg = 'Mixing dicts with non-Series may lead to ambiguous ordering.'
    with pytest.raises(ValueError, match=msg):
        DataFrame({'A': {'a': 'a', 'b': 'b'}, 'B': ['a', 'b', 'c']})
    msg = 'Shape of passed values is \\(4, 3\\), indices imply \\(3, 3\\)'
    with pytest.raises(ValueError, match=msg):
        DataFrame(np.arange(12).reshape((4, 3)), columns=['foo', 'bar', 'baz'], index=date_range('2000-01-01', periods=3))
    arr = np.array([[4, 5, 6]])
    msg = 'Shape of passed values is \\(1, 3\\), indices imply \\(1, 4\\)'
    with pytest.raises(ValueError, match=msg):
        DataFrame(index=[0], columns=range(4), data=arr)
    arr = np.array([4, 5, 6])
    msg = 'Shape of passed values is \\(3, 1\\), indices imply \\(1, 4\\)'
    with pytest.raises(ValueError, match=msg):
        DataFrame(index=[0], columns=range(4), data=arr)
    with pytest.raises(ValueError, match='Must pass 2-d input'):
        DataFrame(np.zeros((3, 3, 3)), columns=['A', 'B', 'C'], index=[1])
    msg = 'Shape of passed values is \\(2, 3\\), indices imply \\(1, 3\\)'
    with pytest.raises(ValueError, match=msg):
        DataFrame(np.random.default_rng(2).random((2, 3)), columns=['A', 'B', 'C'], index=[1])
    msg = 'Shape of passed values is \\(2, 3\\), indices imply \\(2, 2\\)'
    with pytest.raises(ValueError, match=msg):
        DataFrame(np.random.default_rng(2).random((2, 3)), columns=['A', 'B'], index=[1, 2])
    msg = '2 columns passed, passed data had 10 columns'
    with pytest.raises(ValueError, match=msg):
        DataFrame((range(10), range(10, 20)), columns=('ones', 'twos'))
    msg = 'If using all scalar values, you must pass an index'
    with pytest.raises(ValueError, match=msg):
        DataFrame({'a': False, 'b': True})