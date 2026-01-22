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
def test_constructor_series_nonexact_categoricalindex(self):
    ser = Series(range(100))
    ser1 = cut(ser, 10).value_counts().head(5)
    ser2 = cut(ser, 10).value_counts().tail(5)
    result = DataFrame({'1': ser1, '2': ser2})
    index = CategoricalIndex([Interval(-0.099, 9.9, closed='right'), Interval(9.9, 19.8, closed='right'), Interval(19.8, 29.7, closed='right'), Interval(29.7, 39.6, closed='right'), Interval(39.6, 49.5, closed='right'), Interval(49.5, 59.4, closed='right'), Interval(59.4, 69.3, closed='right'), Interval(69.3, 79.2, closed='right'), Interval(79.2, 89.1, closed='right'), Interval(89.1, 99, closed='right')], ordered=True)
    expected = DataFrame({'1': [10] * 5 + [np.nan] * 5, '2': [np.nan] * 5 + [10] * 5}, index=index)
    tm.assert_frame_equal(expected, result)