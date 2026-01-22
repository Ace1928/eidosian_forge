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
def test_groupby_observed_with_agg():
    df = pd.DataFrame({'cat_1': pd.Categorical(list('AB'), categories=list('ABCDE')), 'cat_2': pd.Categorical([1, 2], categories=[1, 2, 3]), 'value_1': np.random.uniform(size=2)})
    expected = df.groupby(['cat_1', 'cat_2'], observed=True).agg('sum')
    ddf = dd.from_pandas(df, 2)
    actual = ddf.groupby(['cat_1', 'cat_2'], observed=True).agg('sum')
    assert_eq(expected, actual)