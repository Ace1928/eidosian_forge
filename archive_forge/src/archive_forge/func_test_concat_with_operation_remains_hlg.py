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
def test_concat_with_operation_remains_hlg():
    pdf1 = pd.DataFrame({'x': [1, 2, 3, 4, 6, 7], 'y': list('abcdef')}, index=[1, 2, 3, 4, 6, 7])
    ddf1 = dd.from_pandas(pdf1, 2)
    pdf2 = pd.DataFrame({'y': list('abcdef')}, index=[8, 9, 10, 11, 12, 13])
    ddf2 = dd.from_pandas(pdf2, 2)
    pdf2['x'] = range(len(pdf2['y']))
    ddf2['x'] = ddf2.assign(partition_count=1).partition_count.cumsum() - 1
    kwargs = {'sort': False}
    expected = pd.concat([pdf1, pdf2], **kwargs)
    result = dd.concat([ddf1, ddf2], **kwargs)
    if not DASK_EXPR_ENABLED:
        assert not hlg_layer_topological(result.dask, 2).is_materialized()
    assert_eq(result, expected)