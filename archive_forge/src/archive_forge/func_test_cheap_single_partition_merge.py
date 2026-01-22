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
@pytest.mark.parametrize('flip', [False, True])
def test_cheap_single_partition_merge(flip):
    a = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6], 'y': list('abdabd')}, index=[10, 20, 30, 40, 50, 60])
    aa = dd.from_pandas(a, npartitions=3)
    b = pd.DataFrame({'x': [1, 2, 3, 4], 'z': list('abda')})
    bb = dd.from_pandas(b, npartitions=1, sort=False)
    pd_inputs = (b, a) if flip else (a, b)
    inputs = (bb, aa) if flip else (aa, bb)
    cc = dd.merge(*inputs, on='x', how='inner')
    if not DASK_EXPR_ENABLED:
        assert not hlg_layer_topological(cc.dask, -1).is_materialized()
    assert all(('shuffle' not in k[0] for k in cc.dask))
    if not DASK_EXPR_ENABLED:
        input_layers = aa.dask.layers.keys() | bb.dask.layers.keys()
        output_layers = cc.dask.layers.keys()
        assert len(output_layers - input_layers) == 1
    list_eq(cc, pd.merge(*pd_inputs, on='x', how='inner'))