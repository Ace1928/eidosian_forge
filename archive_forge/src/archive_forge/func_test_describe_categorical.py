import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_describe_categorical(self):
    df = DataFrame({'value': np.random.default_rng(2).integers(0, 10000, 100)})
    labels = [f'{i} - {i + 499}' for i in range(0, 10000, 500)]
    cat_labels = Categorical(labels, labels)
    df = df.sort_values(by=['value'], ascending=True)
    df['value_group'] = pd.cut(df.value, range(0, 10500, 500), right=False, labels=cat_labels)
    cat = df
    result = cat.describe()
    assert len(result.columns) == 1
    cat = Categorical(['a', 'b', 'b', 'b'], categories=['a', 'b', 'c'], ordered=True)
    s = Series(cat)
    result = s.describe()
    expected = Series([4, 2, 'b', 3], index=['count', 'unique', 'top', 'freq'])
    tm.assert_series_equal(result, expected)
    cat = Series(Categorical(['a', 'b', 'c', 'c']))
    df3 = DataFrame({'cat': cat, 's': ['a', 'b', 'c', 'c']})
    result = df3.describe()
    tm.assert_numpy_array_equal(result['cat'].values, result['s'].values)