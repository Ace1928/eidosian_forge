import re
import sys
import numpy as np
import pytest
from pandas.compat import PYPY
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
def test_min_max_reduce(self):
    cat = Categorical(['a', 'b', 'c', 'd'], ordered=True)
    df = DataFrame(cat)
    result_max = df.agg('max')
    expected_max = Series(Categorical(['d'], dtype=cat.dtype))
    tm.assert_series_equal(result_max, expected_max)
    result_min = df.agg('min')
    expected_min = Series(Categorical(['a'], dtype=cat.dtype))
    tm.assert_series_equal(result_min, expected_min)