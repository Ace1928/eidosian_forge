import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_eval_resolvers_combined(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), columns=list('ab'))
    dict1 = {'c': 2}
    result = df.eval('a + b * c', resolvers=[dict1])
    expected = df['a'] + df['b'] * dict1['c']
    tm.assert_series_equal(result, expected)