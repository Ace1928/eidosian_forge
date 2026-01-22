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
def test_get_dummies_errors():
    with pytest.raises(NotImplementedError):
        s = pd.Series([1, 1, 1, 2, 2, 1, 3, 4])
        ds = dd.from_pandas(s, 2)
        dd.get_dummies(ds)
    df = pd.DataFrame({'x': list('abcbc'), 'y': list('bcbcb')})
    ddf = dd.from_pandas(df, npartitions=2)
    ddf = ddf.astype('category')
    with pytest.raises(NotImplementedError):
        dd.get_dummies(ddf)
    with pytest.raises(NotImplementedError):
        dd.get_dummies(ddf, columns=['x', 'y'])
    with pytest.raises(NotImplementedError):
        dd.get_dummies(ddf.x)