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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='not available')
def test__maybe_align_partitions():
    df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7], 'b': [7, 6, 5, 4, 3, 2, 1]})
    ddf = dd.from_pandas(df + 1, npartitions=2)
    ddf2 = dd.from_pandas(df, npartitions=2)
    a, b = _maybe_align_partitions([ddf, ddf2])
    assert a is ddf
    assert b is ddf2
    ddf = dd.from_pandas(df + 1, npartitions=2, sort=False)
    ddf2 = dd.from_pandas(df, npartitions=2, sort=False)
    assert not ddf.known_divisions
    assert not ddf2.known_divisions
    a, b = _maybe_align_partitions([ddf, ddf2])
    assert a is ddf
    assert b is ddf2
    ddf = dd.from_pandas(df + 1, npartitions=2)
    ddf2 = dd.from_pandas(df, npartitions=3)
    a, b = _maybe_align_partitions([ddf, ddf2])
    assert a.divisions == b.divisions
    ddf = dd.from_pandas(df + 1, npartitions=2, sort=False)
    ddf2 = dd.from_pandas(df, npartitions=3, sort=False)
    assert not ddf.known_divisions
    assert not ddf2.known_divisions
    with pytest.raises(ValueError):
        _maybe_align_partitions([ddf, ddf2])
    ddf = dd.from_pandas(df, npartitions=2)
    ddf2 = dd.from_pandas(df, npartitions=2, sort=False)
    assert not ddf2.known_divisions
    with pytest.raises(ValueError):
        _maybe_align_partitions([ddf, ddf2])