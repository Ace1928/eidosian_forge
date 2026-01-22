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
def test_sequential_joins():
    df1 = pd.DataFrame({'key': list(range(6)), 'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})
    df2 = pd.DataFrame({'key': list(range(4)), 'B': ['B0', 'B1', 'B2', 'B3']})
    df3 = pd.DataFrame({'key': list(range(1, 5)), 'C': ['C0', 'C1', 'C2', 'C3']})
    join_pd = df1.join(df2, how='inner', lsuffix='_l', rsuffix='_r')
    multi_join_pd = join_pd.join(df3, how='inner', lsuffix='_l', rsuffix='_r')
    ddf1 = dd.from_pandas(df1, npartitions=3)
    ddf2 = dd.from_pandas(df2, npartitions=2)
    ddf3 = dd.from_pandas(df3, npartitions=2)
    join_dd = ddf1.join(ddf2, how='inner', lsuffix='_l', rsuffix='_r')
    multi_join_dd = join_dd.join(ddf3, how='inner', lsuffix='_l', rsuffix='_r')
    assert_eq(multi_join_pd, multi_join_dd)