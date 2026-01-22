from itertools import chain
import operator
import numpy as np
import pytest
from pandas._libs.algos import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
def test_rank_int(self, ser, results):
    method, exp = results
    s = ser.dropna().astype('i8')
    result = s.rank(method=method)
    expected = Series(exp).dropna()
    expected.index = result.index
    tm.assert_series_equal(result, expected)