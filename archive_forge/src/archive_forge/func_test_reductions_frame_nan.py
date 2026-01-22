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
@pytest.mark.parametrize('split_every', [False, 2])
def test_reductions_frame_nan(split_every):
    df = pd.DataFrame({'a': [1, 2, np.nan, 4, 5, 6, 7, 8], 'b': [1, 2, np.nan, np.nan, np.nan, 5, np.nan, np.nan], 'c': [np.nan] * 8})
    ddf = dd.from_pandas(df, 3)
    assert_eq(df.sum(), ddf.sum(split_every=split_every))
    assert_eq(df.prod(), ddf.prod(split_every=split_every))
    assert_eq(df.product(), ddf.product(split_every=split_every))
    assert_eq(df.min(), ddf.min(split_every=split_every))
    assert_eq(df.max(), ddf.max(split_every=split_every))
    assert_eq(df.count(), ddf.count(split_every=split_every))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        assert_eq(df.std(), ddf.std(split_every=split_every))
        assert_eq(df.var(), ddf.var(split_every=split_every))
        assert_eq(df.sem(), ddf.sem(split_every=split_every))
        assert_eq(df.std(ddof=0), ddf.std(ddof=0, split_every=split_every))
        assert_eq(df.var(ddof=0), ddf.var(ddof=0, split_every=split_every))
        assert_eq(df.sem(ddof=0), ddf.sem(ddof=0, split_every=split_every))
    assert_eq(df.mean(), ddf.mean(split_every=split_every))
    with warnings.catch_warnings(record=True):
        assert_eq(df.sum(skipna=False), ddf.sum(skipna=False, split_every=split_every))
        assert_eq(df.prod(skipna=False), ddf.prod(skipna=False, split_every=split_every))
        assert_eq(df.product(skipna=False), ddf.product(skipna=False, split_every=split_every))
        assert_eq(df.min(skipna=False), ddf.min(skipna=False, split_every=split_every))
        assert_eq(df.max(skipna=False), ddf.max(skipna=False, split_every=split_every))
        assert_eq(df.std(skipna=False), ddf.std(skipna=False, split_every=split_every))
        assert_eq(df.var(skipna=False), ddf.var(skipna=False, split_every=split_every))
        assert_eq(df.sem(skipna=False), ddf.sem(skipna=False, split_every=split_every))
        assert_eq(df.std(skipna=False, ddof=0), ddf.std(skipna=False, ddof=0, split_every=split_every))
        assert_eq(df.var(skipna=False, ddof=0), ddf.var(skipna=False, ddof=0, split_every=split_every))
        assert_eq(df.sem(skipna=False, ddof=0), ddf.sem(skipna=False, ddof=0, split_every=split_every))
        assert_eq(df.mean(skipna=False), ddf.mean(skipna=False, split_every=split_every))
        assert_eq(df.sum(axis=1, skipna=False), ddf.sum(axis=1, skipna=False, split_every=split_every))
        assert_eq(df.prod(axis=1, skipna=False), ddf.prod(axis=1, skipna=False, split_every=split_every))
        assert_eq(df.product(axis=1, skipna=False), ddf.product(axis=1, skipna=False, split_every=split_every))
        assert_eq(df.min(axis=1, skipna=False), ddf.min(axis=1, skipna=False, split_every=split_every))
        assert_eq(df.max(axis=1, skipna=False), ddf.max(axis=1, skipna=False, split_every=split_every))
        assert_eq(df.std(axis=1, skipna=False), ddf.std(axis=1, skipna=False, split_every=split_every))
        assert_eq(df.var(axis=1, skipna=False), ddf.var(axis=1, skipna=False, split_every=split_every))
        assert_eq(df.sem(axis=1, skipna=False), ddf.sem(axis=1, skipna=False, split_every=split_every))
        assert_eq(df.std(axis=1, skipna=False, ddof=0), ddf.std(axis=1, skipna=False, ddof=0, split_every=split_every))
        assert_eq(df.var(axis=1, skipna=False, ddof=0), ddf.var(axis=1, skipna=False, ddof=0, split_every=split_every))
        assert_eq(df.sem(axis=1, skipna=False, ddof=0), ddf.sem(axis=1, skipna=False, ddof=0, split_every=split_every))
        assert_eq(df.mean(axis=1, skipna=False), ddf.mean(axis=1, skipna=False, split_every=split_every))