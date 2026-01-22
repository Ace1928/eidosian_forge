import datetime
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_operators_na_handling(self):
    ser = Series(['foo', 'bar', 'baz', np.nan])
    result = 'prefix_' + ser
    expected = Series(['prefix_foo', 'prefix_bar', 'prefix_baz', np.nan])
    tm.assert_series_equal(result, expected)
    result = ser + '_suffix'
    expected = Series(['foo_suffix', 'bar_suffix', 'baz_suffix', np.nan])
    tm.assert_series_equal(result, expected)