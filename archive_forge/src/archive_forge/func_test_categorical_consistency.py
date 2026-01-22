from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from pandas.util import hash_pandas_object
import dask.dataframe as dd
from dask.dataframe import _compat
from dask.dataframe._compat import tm
from dask.dataframe.utils import assert_eq
def test_categorical_consistency():
    for s1 in [pd.Series(['a', 'b', 'c', 'd']), pd.Series([1000, 2000, 3000, 4000]), pd.Series(pd.date_range(0, periods=4))]:
        s2 = s1.astype('category').cat.set_categories(s1)
        s3 = s2.cat.set_categories(list(reversed(s1)))
        for categorize in [True, False]:
            h1 = hash_pandas_object(s1, categorize=categorize)
            h2 = hash_pandas_object(s2, categorize=categorize)
            h3 = hash_pandas_object(s3, categorize=categorize)
            tm.assert_series_equal(h1, h2)
            tm.assert_series_equal(h1, h3)