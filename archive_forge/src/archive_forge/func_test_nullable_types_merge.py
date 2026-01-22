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
@pytest.mark.parametrize('dtype', ['Int64', 'Float64', pytest.param('int64[pyarrow]', marks=pytest.mark.skipif(pa is None or not PANDAS_GE_150, reason='Support for ArrowDtypes requires pyarrow and pandas>=1.5.0')), pytest.param('float64[pyarrow]', marks=pytest.mark.skipif(pa is None or not PANDAS_GE_150, reason='Support for ArrowDtypes requires pyarrow and pandas>=1.5.0'))])
def test_nullable_types_merge(dtype):
    df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [1, 1, 3], 'c': list('aab')})
    df2 = pd.DataFrame({'a': [1, 2, 3], 'e': [1, 1, 3], 'f': list('aab')})
    df1['a'] = df1['a'].astype(dtype)
    df2['a'] = df2['a'].astype(dtype)
    ddf1 = dd.from_pandas(df1, npartitions=2)
    ddf2 = dd.from_pandas(df2, npartitions=2)
    expect = df1.merge(df2, on='a')
    actual = ddf1.merge(ddf2, on='a')
    assert_eq(expect, actual, check_index=False)
    expect = pd.merge(df1, df2, on='a')
    actual = dd.merge(ddf1, ddf2, on='a')
    assert_eq(expect, actual, check_index=False)