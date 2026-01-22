from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from pandas.util import hash_pandas_object
import dask.dataframe as dd
from dask.dataframe import _compat
from dask.dataframe._compat import tm
from dask.dataframe.utils import assert_eq
@pytest.mark.parametrize('obj', [pd.Series([1, 2, 3]), pd.Series([1.0, 1.5, 3.2]), pd.Series([1.0, 1.5, 3.2], index=[1.5, 1.1, 3.3]), pd.Series(['a', 'b', 'c']), pd.Series([True, False, True]), pd.Index([1, 2, 3]), pd.Index([True, False, True]), pd.DataFrame({'x': ['a', 'b', 'c'], 'y': [1, 2, 3]}), _compat.makeMissingDataframe(), _compat.makeMixedDataFrame(), _compat.makeTimeDataFrame(), _compat.makeTimeSeries(), _compat.makeTimedeltaIndex()])
def test_hash_pandas_object(obj):
    a = hash_pandas_object(obj)
    b = hash_pandas_object(obj)
    if isinstance(a, np.ndarray):
        np.testing.assert_equal(a, b)
    else:
        assert_eq(a, b)