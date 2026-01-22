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
def test_groupby_agg_custom__name_clash_with_internal_same_column():
    """for a single input column only unique names are allowed"""
    d = pd.DataFrame({'g': [0, 0, 1] * 3, 'b': [1, 2, 3] * 3})
    a = dd.from_pandas(d, npartitions=2)
    agg_func = dd.Aggregation('sum', lambda s: s.sum(), lambda s0: s0.sum())
    with pytest.raises(ValueError):
        a.groupby('g').aggregate({'b': [agg_func, 'sum']})