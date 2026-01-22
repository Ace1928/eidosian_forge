from __future__ import annotations
import contextlib
import operator
import warnings
from datetime import datetime
from functools import partial
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.dataframe import _compat
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.backends import grouper_dispatch
from dask.dataframe.groupby import NUMERIC_ONLY_NOT_IMPLEMENTED
from dask.dataframe.utils import assert_dask_graph, assert_eq, pyarrow_strings_enabled
from dask.utils import M
from dask.utils_test import _check_warning, hlg_layer
@pytest.mark.parametrize('sort', [True, False])
def test_groupby_dropna_with_agg(sort):
    df = pd.DataFrame({'id1': ['a', None, 'b'], 'id2': [1, 2, None], 'v1': [4.5, 5.5, None]})
    if PANDAS_GE_200:
        expected = df.groupby(['id1', 'id2'], dropna=False, sort=sort).agg('sum')
    else:
        expected = df.groupby(['id1', 'id2'], dropna=False, sort=True).agg('sum')
    ddf = dd.from_pandas(df, 1)
    actual = ddf.groupby(['id1', 'id2'], dropna=False, sort=sort).agg('sum')
    assert_eq(expected, actual)