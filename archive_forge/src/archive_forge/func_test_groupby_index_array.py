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
@pytest.mark.xfail(DASK_EXPR_ENABLED, reason='grouper does not have divisions and groupby aligns')
def test_groupby_index_array():
    df = _compat.makeTimeDataFrame()
    ddf = dd.from_pandas(df, npartitions=2)
    assert_eq(df.A.groupby(df.index.month).nunique(), ddf.A.groupby(ddf.index.month).nunique(), check_names=False)
    assert_eq(df.groupby(df.index.month).A.nunique(), ddf.groupby(ddf.index.month).A.nunique(), check_names=False)