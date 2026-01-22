from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
def test_value_counts_dropna(self):
    tm.assert_series_equal(Series([True, True, False]).value_counts(dropna=True), Series([2, 1], index=[True, False], name='count'))
    tm.assert_series_equal(Series([True, True, False]).value_counts(dropna=False), Series([2, 1], index=[True, False], name='count'))
    tm.assert_series_equal(Series([True] * 3 + [False] * 2 + [None] * 5).value_counts(dropna=True), Series([3, 2], index=Index([True, False], dtype=object), name='count'))
    tm.assert_series_equal(Series([True] * 5 + [False] * 3 + [None] * 2).value_counts(dropna=False), Series([5, 3, 2], index=[True, False, None], name='count'))
    tm.assert_series_equal(Series([10.3, 5.0, 5.0]).value_counts(dropna=True), Series([2, 1], index=[5.0, 10.3], name='count'))
    tm.assert_series_equal(Series([10.3, 5.0, 5.0]).value_counts(dropna=False), Series([2, 1], index=[5.0, 10.3], name='count'))
    tm.assert_series_equal(Series([10.3, 5.0, 5.0, None]).value_counts(dropna=True), Series([2, 1], index=[5.0, 10.3], name='count'))
    result = Series([10.3, 10.3, 5.0, 5.0, 5.0, None]).value_counts(dropna=False)
    expected = Series([3, 2, 1], index=[5.0, 10.3, None], name='count')
    tm.assert_series_equal(result, expected)