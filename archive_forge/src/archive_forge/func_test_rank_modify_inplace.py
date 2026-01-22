from itertools import chain
import operator
import numpy as np
import pytest
from pandas._libs.algos import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
def test_rank_modify_inplace(self):
    s = Series([Timestamp('2017-01-05 10:20:27.569000'), NaT])
    expected = s.copy()
    s.rank()
    result = s
    tm.assert_series_equal(result, expected)