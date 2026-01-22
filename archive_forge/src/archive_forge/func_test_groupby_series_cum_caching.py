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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='MIDX columns not supported')
def test_groupby_series_cum_caching():
    """Test caching behavior of cumulative operations on grouped Series

    Relates to #3755
    """
    df = pd.DataFrame(dict(a=list('aabbcc')), index=pd.date_range(start='20100101', periods=6))
    df['ones'] = 1
    df['twos'] = 2
    ops = ['cumsum', 'cumprod']
    for op in ops:
        ddf = dd.from_pandas(df, npartitions=3)
        dcum = ddf.groupby(['a'])
        res0_a, res1_a = dask.compute(getattr(dcum['ones'], op)(), getattr(dcum['twos'], op)())
        cum = df.groupby(['a'])
        res0_b, res1_b = (getattr(cum['ones'], op)(), getattr(cum['twos'], op)())
        assert res0_a.equals(res0_b)
        assert res1_a.equals(res1_b)