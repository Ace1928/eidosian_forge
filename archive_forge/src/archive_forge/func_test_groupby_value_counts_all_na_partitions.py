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
def test_groupby_value_counts_all_na_partitions():
    size = 100
    na_size = 90
    npartitions = 10
    df = pd.DataFrame({'A': np.random.randint(0, 2, size=size, dtype=bool), 'B': np.append(np.nan * np.zeros(na_size), np.random.randn(size - na_size))})
    ddf = dd.from_pandas(df, npartitions=npartitions)
    assert_eq(ddf.groupby('A')['B'].value_counts(), df.groupby('A')['B'].value_counts())