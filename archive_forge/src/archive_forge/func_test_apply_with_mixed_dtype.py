from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_with_mixed_dtype():
    df = DataFrame({'foo1': np.random.default_rng(2).standard_normal(6), 'foo2': ['one', 'two', 'two', 'three', 'one', 'two']})
    result = df.apply(lambda x: x, axis=1).dtypes
    expected = df.dtypes
    tm.assert_series_equal(result, expected)
    df = DataFrame({'c1': [1, 2, 6, 6, 8]})
    df['c2'] = df.c1 / 2.0
    result1 = df.groupby('c2').mean().reset_index().c2
    result2 = df.groupby('c2', as_index=False).mean().c2
    tm.assert_series_equal(result1, result2)