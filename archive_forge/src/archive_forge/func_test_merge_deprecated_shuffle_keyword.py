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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='not deprecated in dask-expr')
def test_merge_deprecated_shuffle_keyword(shuffle_method):
    A = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6], 'y': [1, 1, 2, 2, 3, 4]})
    a = dd.repartition(A, [0, 4, 5])
    B = pd.DataFrame({'y': [1, 3, 4, 4, 5, 6], 'z': [6, 5, 4, 3, 2, 1]})
    b = dd.repartition(B, [0, 2, 5])
    expected = pd.merge(A, B, left_index=True, right_index=True)
    with pytest.warns(FutureWarning, match="'shuffle' keyword is deprecated"):
        result = dd.merge(a, b, left_index=True, right_index=True, shuffle=shuffle_method)
    assert_eq(result, expected)
    with pytest.warns(FutureWarning, match="'shuffle' keyword is deprecated"):
        result = a.merge(b, left_index=True, right_index=True, shuffle=shuffle_method)
    assert_eq(result, expected)