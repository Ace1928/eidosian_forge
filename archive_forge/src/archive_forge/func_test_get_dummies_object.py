from __future__ import annotations
import contextlib
import warnings
import numpy as np
import pandas as pd
import pytest
from packaging.version import parse as parse_version
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_VERSION, tm
from dask.dataframe.reshape import _get_dummies_dtype_default
from dask.dataframe.utils import assert_eq
def test_get_dummies_object():
    df = pd.DataFrame({'a': pd.Categorical([1, 2, 3, 4, 4, 3, 2, 1]), 'b': list('abcdabcd'), 'c': pd.Categorical(list('abcdabcd'))})
    ddf = dd.from_pandas(df, 2)
    exp = pd.get_dummies(df, columns=['a', 'c'])
    res = dd.get_dummies(ddf, columns=['a', 'c'])
    assert_eq(res, exp)
    tm.assert_index_equal(res.columns, exp.columns)
    with pytest.raises(NotImplementedError):
        dd.get_dummies(ddf)
    with pytest.raises(NotImplementedError):
        dd.get_dummies(ddf.b)
    with pytest.raises(NotImplementedError):
        dd.get_dummies(ddf, columns=['b'])