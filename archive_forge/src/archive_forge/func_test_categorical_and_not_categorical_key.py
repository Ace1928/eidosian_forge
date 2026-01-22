import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.common import ensure_platform_int
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_categorical_and_not_categorical_key(observed):
    df_with_categorical = DataFrame({'A': Categorical(['a', 'b', 'a'], categories=['a', 'b', 'c']), 'B': [1, 2, 3], 'C': ['a', 'b', 'a']})
    df_without_categorical = DataFrame({'A': ['a', 'b', 'a'], 'B': [1, 2, 3], 'C': ['a', 'b', 'a']})
    result = df_with_categorical.groupby(['A', 'C'], observed=observed).transform('sum')
    expected = df_without_categorical.groupby(['A', 'C']).transform('sum')
    tm.assert_frame_equal(result, expected)
    expected_explicit = DataFrame({'B': [4, 2, 4]})
    tm.assert_frame_equal(result, expected_explicit)
    result = df_with_categorical.groupby(['A', 'C'], observed=observed)['B'].transform('sum')
    expected = df_without_categorical.groupby(['A', 'C'])['B'].transform('sum')
    tm.assert_series_equal(result, expected)
    expected_explicit = Series([4, 2, 4], name='B')
    tm.assert_series_equal(result, expected_explicit)