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
def test_merge_indexed_dataframe_to_indexed_dataframe():
    A = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6]}, index=[1, 2, 3, 4, 6, 7])
    a = dd.repartition(A, [1, 4, 7])
    B = pd.DataFrame({'y': list('abcdef')}, index=[1, 2, 4, 5, 6, 8])
    b = dd.repartition(B, [1, 2, 5, 8])
    if DASK_EXPR_ENABLED:
        c = a.merge(b, how='left')
    else:
        c = merge_indexed_dataframes(a, b, how='left')
    assert c.divisions[0] == a.divisions[0]
    assert c.divisions[-1] == max(a.divisions + b.divisions)
    assert_eq(c, A.join(B))
    if DASK_EXPR_ENABLED:
        c = a.merge(b, how='right')
    else:
        c = merge_indexed_dataframes(a, b, how='right')
    assert c.divisions[0] == b.divisions[0]
    assert c.divisions[-1] == b.divisions[-1]
    assert_eq(c, A.join(B, how='right'))
    if DASK_EXPR_ENABLED:
        c = a.merge(b, how='inner')
    else:
        c = merge_indexed_dataframes(a, b, how='inner')
    assert c.divisions[0] == 1
    assert c.divisions[-1] == max(a.divisions + b.divisions)
    assert_eq(c.compute(), A.join(B, how='inner'))
    if DASK_EXPR_ENABLED:
        c = a.merge(b, how='outer')
    else:
        c = merge_indexed_dataframes(a, b, how='outer')
    assert c.divisions[0] == 1
    assert c.divisions[-1] == 8
    assert_eq(c.compute(), A.join(B, how='outer'))
    if not DASK_EXPR_ENABLED:
        assert sorted(merge_indexed_dataframes(a, b, how='inner').dask) == sorted(merge_indexed_dataframes(a, b, how='inner').dask)
        assert sorted(merge_indexed_dataframes(a, b, how='inner').dask) != sorted(merge_indexed_dataframes(a, b, how='outer').dask)