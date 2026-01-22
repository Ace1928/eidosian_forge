from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
@pytest.mark.parametrize('ops, by_row, expected', [([lambda x: x + 1], 'compat', DataFrame({('a', '<lambda>'): [2, 3]})), ([lambda x: x + 1], False, DataFrame({('a', '<lambda>'): [2, 3]})), ([lambda x: x.sum()], 'compat', DataFrame({'a': [3]}, index=['<lambda>'])), ([lambda x: x.sum()], False, DataFrame({'a': [3]}, index=['<lambda>'])), (['sum', np.sum, lambda x: x.sum()], 'compat', DataFrame({'a': [3, 3, 3]}, index=['sum', 'sum', '<lambda>'])), (['sum', np.sum, lambda x: x.sum()], False, DataFrame({'a': [3, 3, 3]}, index=['sum', 'sum', '<lambda>'])), ([lambda x: x + 1, lambda x: 3], 'compat', DataFrame([[2, 3], [3, 3]], columns=[['a', 'a'], ['<lambda>', '<lambda>']])), ([lambda x: 2, lambda x: 3], False, DataFrame({'a': [2, 3]}, ['<lambda>', '<lambda>']))])
def test_listlike_lambda(ops, by_row, expected):
    df = DataFrame({'a': [1, 2]})
    result = df.apply(ops, by_row=by_row)
    tm.assert_equal(result, expected)