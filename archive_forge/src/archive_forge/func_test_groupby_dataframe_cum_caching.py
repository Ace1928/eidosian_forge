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
@pytest.mark.parametrize('op', ['cumsum', 'cumprod'])
def test_groupby_dataframe_cum_caching(op):
    """Test caching behavior of cumulative operations on grouped dataframes.

    Relates to #3756.
    """
    df = pd.DataFrame(dict(a=list('aabbcc')), index=pd.date_range(start='20100101', periods=6))
    df['ones'] = 1
    df['twos'] = 2
    ddf = dd.from_pandas(df, npartitions=3)
    ddf0 = getattr(ddf.groupby(['a']), op)()
    ddf1 = ddf.rename(columns={'ones': 'foo', 'twos': 'bar'})
    ddf1 = getattr(ddf1.groupby(['a']), op)()
    res0_a, res1_a = dask.compute(ddf0, ddf1)
    res0_b, res1_b = (ddf0.compute(), ddf1.compute())
    assert res0_a.equals(res0_b)
    assert res1_a.equals(res1_b)