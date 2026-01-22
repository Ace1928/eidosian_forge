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
@pytest.mark.parametrize('how', ['left', 'outer'])
@pytest.mark.parametrize('npartitions_base', [1, 2, 3])
@pytest.mark.parametrize('npartitions_other', [1, 2, 3])
def test_pairwise_merge_results_in_identical_output_df(how, npartitions_base, npartitions_other):
    dfs_to_merge = []
    for i in range(10):
        df = pd.DataFrame({f'{i}A': [5, 6, 7, 8], f'{i}B': [4, 3, 2, 1]}, index=[0, 1, 2, 3])
        ddf = dd.from_pandas(df, npartitions_other)
        dfs_to_merge.append(ddf)
    ddf_loop = dd.from_pandas(pd.DataFrame(index=[0, 1, 3]), npartitions_base)
    for ddf in dfs_to_merge:
        ddf_loop = ddf_loop.join(ddf, how=how)
    ddf_pairwise = dd.from_pandas(pd.DataFrame(index=[0, 1, 3]), npartitions_base)
    ddf_pairwise = ddf_pairwise.join(dfs_to_merge, how=how)
    assert_eq(ddf_pairwise, ddf_loop)