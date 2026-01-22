from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_groupby_dict_mapping(self):
    s = Series({'T1': 5})
    result = s.groupby({'T1': 'T2'}).agg('sum')
    expected = s.groupby(['T2']).agg('sum')
    tm.assert_series_equal(result, expected)
    s = Series([1.0, 2.0, 3.0, 4.0], index=list('abcd'))
    mapping = {'a': 0, 'b': 0, 'c': 1, 'd': 1}
    result = s.groupby(mapping).mean()
    result2 = s.groupby(mapping).agg('mean')
    exp_key = np.array([0, 0, 1, 1], dtype=np.int64)
    expected = s.groupby(exp_key).mean()
    expected2 = s.groupby(exp_key).mean()
    tm.assert_series_equal(result, expected)
    tm.assert_series_equal(result, result2)
    tm.assert_series_equal(result, expected2)