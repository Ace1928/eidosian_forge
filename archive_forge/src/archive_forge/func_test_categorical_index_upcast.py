from datetime import datetime
import numpy as np
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_categorical_index_upcast(self):
    a = DataFrame({'foo': [1, 2]}, index=Categorical(['foo', 'bar']))
    b = DataFrame({'foo': [4, 3]}, index=Categorical(['baz', 'bar']))
    res = pd.concat([a, b])
    exp = DataFrame({'foo': [1, 2, 4, 3]}, index=['foo', 'bar', 'baz', 'bar'])
    tm.assert_equal(res, exp)
    a = Series([1, 2], index=Categorical(['foo', 'bar']))
    b = Series([4, 3], index=Categorical(['baz', 'bar']))
    res = pd.concat([a, b])
    exp = Series([1, 2, 4, 3], index=['foo', 'bar', 'baz', 'bar'])
    tm.assert_equal(res, exp)