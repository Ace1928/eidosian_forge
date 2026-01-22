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
@pytest.mark.parametrize('by', ['foo', ['foo', 'bar']])
@pytest.mark.parametrize('int_dtype', ['uint8', 'int32', 'int64'])
def test_groupby_value_counts(by, int_dtype):
    rng = np.random.RandomState(42)
    df = pd.DataFrame({'foo': rng.randint(3, size=100), 'bar': rng.randint(4, size=100), 'baz': rng.randint(5, size=100)}, dtype=int_dtype)
    ddf = dd.from_pandas(df, npartitions=2)
    pd_gb = df.groupby(by).baz.value_counts()
    dd_gb = ddf.groupby(by).baz.value_counts()
    assert_eq(dd_gb, pd_gb)