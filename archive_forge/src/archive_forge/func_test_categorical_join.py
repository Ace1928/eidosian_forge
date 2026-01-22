from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
from pandas.api.types import is_object_dtype
import dask.dataframe as dd
from dask._compatibility import PY_VERSION
from dask.base import compute_as_if_collection
from dask.dataframe._compat import (
from dask.dataframe.core import _Frame
from dask.dataframe.methods import concat
from dask.dataframe.multi import (
from dask.dataframe.utils import (
from dask.utils_test import hlg_layer, hlg_layer_topological
def test_categorical_join():
    df = pd.DataFrame({'join_col': ['a', 'a', 'b', 'b'], 'a': [0, 0, 10, 10]})
    df2 = pd.DataFrame({'b': [1, 2, 1, 2]}, index=['a', 'a', 'b', 'b'])
    ddf = dd.from_pandas(df, npartitions=2)
    ddf2 = dd.from_pandas(df2, npartitions=1)
    ddf['join_col'] = ddf['join_col'].astype('category')
    ddf2.index = ddf2.index.astype('category')
    expected = ddf.compute().join(ddf2.compute(), on='join_col', how='left')
    actual_dask = ddf.join(ddf2, on='join_col', how='left')
    assert actual_dask.join_col.dtype == 'category'
    actual = actual_dask.compute()
    assert actual.join_col.dtype == 'category'
    assert assert_eq(expected, actual)