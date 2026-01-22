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
@pytest.mark.parametrize('join', ['inner', 'outer'])
def test_indexed_concat(join):
    A = pd.DataFrame({'x': [1, 2, 3, 4, 6, 7], 'y': list('abcdef')}, index=[1, 2, 3, 4, 6, 7])
    a = dd.repartition(A, [1, 4, 7])
    B = pd.DataFrame({'x': [10, 20, 40, 50, 60, 80]}, index=[1, 2, 4, 5, 6, 8])
    b = dd.repartition(B, [1, 2, 5, 8])
    expected = pd.concat([A, B], axis=0, join=join, sort=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        if DASK_EXPR_ENABLED:
            result = dd.concat([a, b], join=join)
        else:
            result = concat_indexed_dataframes([a, b], join=join)
            assert sorted(concat_indexed_dataframes([a, b], join=join).dask) == sorted(concat_indexed_dataframes([a, b], join=join).dask)
            assert sorted(concat_indexed_dataframes([a, b], join='inner').dask) != sorted(concat_indexed_dataframes([a, b], join='outer').dask)
        assert_eq(result, expected)