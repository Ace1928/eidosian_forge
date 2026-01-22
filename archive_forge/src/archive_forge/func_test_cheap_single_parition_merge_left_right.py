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
@pytest.mark.parametrize('how', ['left', 'right'])
@pytest.mark.parametrize('flip', [False, True])
def test_cheap_single_parition_merge_left_right(how, flip):
    a = pd.DataFrame({'x': range(8), 'z': list('ababbdda')}, index=range(8))
    aa = dd.from_pandas(a, npartitions=1)
    b = pd.DataFrame({'x': [1, 2, 3, 4], 'z': list('abda')}, index=range(4))
    bb = dd.from_pandas(b, npartitions=1)
    pd_inputs = (b, a) if flip else (a, b)
    inputs = (bb, aa) if flip else (aa, bb)
    actual = dd.merge(*inputs, left_index=True, right_on='x', how=how)
    expected = pd.merge(*pd_inputs, left_index=True, right_on='x', how=how)
    if not DASK_EXPR_ENABLED:
        assert not hlg_layer_topological(actual.dask, -1).is_materialized()
    assert_eq(actual, expected)
    actual = dd.merge(*inputs, left_on='x', right_index=True, how=how)
    expected = pd.merge(*pd_inputs, left_on='x', right_index=True, how=how)
    if not DASK_EXPR_ENABLED:
        assert not hlg_layer_topological(actual.dask, -1).is_materialized()
    assert_eq(actual, expected)