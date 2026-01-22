from itertools import chain
import operator
import numpy as np
import pytest
from pandas._libs.algos import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
@pytest.mark.parametrize('dtype', [None, object])
def test_rank_tie_methods(self, ser, results, dtype):
    method, exp = results
    ser = ser if dtype is None else ser.astype(dtype)
    result = ser.rank(method=method)
    tm.assert_series_equal(result, Series(exp))