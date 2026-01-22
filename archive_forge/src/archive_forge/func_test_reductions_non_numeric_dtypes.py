from __future__ import annotations
import contextlib
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_scalar
import dask.dataframe as dd
from dask.array.numpy_compat import NUMPY_GE_125
from dask.dataframe._compat import (
from dask.dataframe.utils import (
@pytest.mark.xfail_with_pyarrow_strings
def test_reductions_non_numeric_dtypes():
    if DASK_EXPR_ENABLED:
        pytest.skip(reason='no arrow strings yet')

    def check_raises(d, p, func):
        pytest.raises((TypeError, ValueError), lambda: getattr(d, func)().compute())
        pytest.raises((TypeError, ValueError), lambda: getattr(p, func)())
    pds = pd.Series(['a', 'b', 'c', 'd', 'e'])
    dds = dd.from_pandas(pds, 2)
    assert_eq(dds.sum(), pds.sum())
    check_raises(dds, pds, 'prod')
    check_raises(dds, pds, 'product')
    assert_eq(dds.min(), pds.min())
    assert_eq(dds.max(), pds.max())
    assert_eq(dds.count(), pds.count())
    check_raises(dds, pds, 'std')
    check_raises(dds, pds, 'var')
    check_raises(dds, pds, 'sem')
    check_raises(dds, pds, 'skew')
    check_raises(dds, pds, 'kurtosis')
    assert_eq(dds.nunique(), pds.nunique())
    for pds in [pd.Series(pd.Categorical([1, 2, 3, 4, 5], ordered=True)), pd.Series(pd.Categorical(list('abcde'), ordered=True)), pd.Series(pd.date_range('2011-01-01', freq='D', periods=5))]:
        dds = dd.from_pandas(pds, 2)
        check_raises(dds, pds, 'sum')
        check_raises(dds, pds, 'prod')
        check_raises(dds, pds, 'product')
        assert_eq(dds.min(), pds.min())
        assert_eq(dds.max(), pds.max())
        assert_eq(dds.count(), pds.count())
        if pds.dtype != 'datetime64[ns]':
            check_raises(dds, pds, 'std')
        check_raises(dds, pds, 'var')
        check_raises(dds, pds, 'sem')
        check_raises(dds, pds, 'skew')
        check_raises(dds, pds, 'kurtosis')
        assert_eq(dds.nunique(), pds.nunique())
    pds = pd.Series(pd.timedelta_range('1 days', freq='D', periods=5))
    dds = dd.from_pandas(pds, 2)
    assert_eq(dds.sum(), pds.sum())
    assert_eq(dds.min(), pds.min())
    assert_eq(dds.max(), pds.max())
    assert_eq(dds.count(), pds.count())
    check_raises(dds, pds, 'skew')
    check_raises(dds, pds, 'kurtosis')
    assert_eq(dds.nunique(), pds.nunique())