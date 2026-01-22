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
@pytest.mark.parametrize('group_keys', [True, False, None])
@pytest.mark.parametrize('limit', [None, 1, 4])
def test_bfill(group_keys, limit):
    df = pd.DataFrame({'A': [1, 1, 2, 2], 'B': [3, 4, 3, 4], 'C': [np.nan, 3, np.nan, np.nan], 'D': [np.nan, 4, np.nan, 5], 'E': [np.nan, 6, np.nan, 7]})
    ddf = dd.from_pandas(df, npartitions=2)
    assert_eq(df.groupby('A', group_keys=group_keys).bfill(limit=limit), ddf.groupby('A', group_keys=group_keys).bfill(limit=limit))
    assert_eq(df.groupby('A', group_keys=group_keys).B.bfill(limit=limit), ddf.groupby('A', group_keys=group_keys).B.bfill(limit=limit))
    assert_eq(df.groupby(['A', 'B'], group_keys=group_keys).bfill(limit=limit), ddf.groupby(['A', 'B'], group_keys=group_keys).bfill(limit=limit))