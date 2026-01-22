import math
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_where_unobserved_categories(self):
    ser = Series(Categorical(['a', 'b', 'c'], categories=['d', 'c', 'b', 'a']))
    result = ser.where([True, True, False], other='b')
    expected = Series(Categorical(['a', 'b', 'b'], categories=ser.cat.categories))
    tm.assert_series_equal(result, expected)