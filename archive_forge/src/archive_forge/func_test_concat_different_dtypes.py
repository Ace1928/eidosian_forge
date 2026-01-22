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
@pytest.mark.xfail(PANDAS_GE_210 or DASK_EXPR_ENABLED, reason='catch_warnings seems flaky', strict=False)
@pytest.mark.parametrize('value_1, value_2', [(1.0, 1), (1.0, 'one'), (1.0, pd.to_datetime('1970-01-01')), (1, 'one'), (1, pd.to_datetime('1970-01-01')), ('one', pd.to_datetime('1970-01-01'))])
def test_concat_different_dtypes(value_1, value_2):
    df_1 = pd.DataFrame({'x': [value_1]})
    df_2 = pd.DataFrame({'x': [value_2]})
    df = pd.concat([df_1, df_2], axis=0)
    expected_dtype = get_string_dtype() if is_object_dtype(df['x'].dtype) else df['x'].dtype
    ddf_1 = dd.from_pandas(df_1, npartitions=1)
    ddf_2 = dd.from_pandas(df_2, npartitions=1)
    ddf = dd.concat([ddf_1, ddf_2], axis=0)
    dask_dtypes = list(ddf.map_partitions(lambda x: x.dtypes).compute())
    assert dask_dtypes == [expected_dtype, expected_dtype]