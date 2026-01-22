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
@pytest.mark.parametrize('empty', [True, False])
def test_split_apply_combine_on_series(empty):
    if empty:
        pdf = pd.DataFrame({'a': [1.0], 'b': [1.0]}, index=[0]).iloc[:0]
        ddofs = []
    else:
        ddofs = [0, 1, 2]
        pdf = pd.DataFrame({'a': [1, 2, 6, 4, 4, 6, 4, 3, 7], 'b': [4, 2, 7, 3, 3, 1, 1, 1, 2]}, index=[0, 1, 3, 5, 6, 8, 9, 9, 9])
    ddf = dd.from_pandas(pdf, npartitions=3)
    for ddkey, pdkey in [('b', 'b'), (ddf.b, pdf.b), (ddf.b + 1, pdf.b + 1)]:
        assert_eq(ddf.groupby(ddkey).a.min(), pdf.groupby(pdkey).a.min())
        assert_eq(ddf.groupby(ddkey).a.max(), pdf.groupby(pdkey).a.max())
        assert_eq(ddf.groupby(ddkey).a.count(), pdf.groupby(pdkey).a.count())
        assert_eq(ddf.groupby(ddkey).a.mean(), pdf.groupby(pdkey).a.mean())
        assert_eq(ddf.groupby(ddkey).a.nunique(), pdf.groupby(pdkey).a.nunique())
        assert_eq(ddf.groupby(ddkey).a.size(), pdf.groupby(pdkey).a.size())
        assert_eq(ddf.groupby(ddkey).a.first(), pdf.groupby(pdkey).a.first())
        assert_eq(ddf.groupby(ddkey).a.last(), pdf.groupby(pdkey).a.last())
        assert_eq(ddf.groupby(ddkey).a.tail(), pdf.groupby(pdkey).a.tail())
        assert_eq(ddf.groupby(ddkey).a.head(), pdf.groupby(pdkey).a.head())
        for ddof in ddofs:
            assert_eq(ddf.groupby(ddkey).a.var(ddof), pdf.groupby(pdkey).a.var(ddof))
            assert_eq(ddf.groupby(ddkey).a.std(ddof), pdf.groupby(pdkey).a.std(ddof))
        assert_eq(ddf.groupby(ddkey).sum(), pdf.groupby(pdkey).sum())
        assert_eq(ddf.groupby(ddkey).min(), pdf.groupby(pdkey).min())
        assert_eq(ddf.groupby(ddkey).max(), pdf.groupby(pdkey).max())
        assert_eq(ddf.groupby(ddkey).count(), pdf.groupby(pdkey).count())
        assert_eq(ddf.groupby(ddkey).mean(), pdf.groupby(pdkey).mean())
        assert_eq(ddf.groupby(ddkey).size(), pdf.groupby(pdkey).size())
        assert_eq(ddf.groupby(ddkey).first(), pdf.groupby(pdkey).first())
        assert_eq(ddf.groupby(ddkey).last(), pdf.groupby(pdkey).last())
        assert_eq(ddf.groupby(ddkey).prod(), pdf.groupby(pdkey).prod())
        for ddof in ddofs:
            assert_eq(ddf.groupby(ddkey).var(ddof), pdf.groupby(pdkey).var(ddof), check_dtype=False)
            assert_eq(ddf.groupby(ddkey).std(ddof), pdf.groupby(pdkey).std(ddof), check_dtype=False)
    for ddkey, pdkey in [(ddf.b, pdf.b), (ddf.b + 1, pdf.b + 1)]:
        assert_eq(ddf.a.groupby(ddkey).sum(), pdf.a.groupby(pdkey).sum(), check_names=False)
        assert_eq(ddf.a.groupby(ddkey).max(), pdf.a.groupby(pdkey).max(), check_names=False)
        assert_eq(ddf.a.groupby(ddkey).count(), pdf.a.groupby(pdkey).count(), check_names=False)
        assert_eq(ddf.a.groupby(ddkey).mean(), pdf.a.groupby(pdkey).mean(), check_names=False)
        assert_eq(ddf.a.groupby(ddkey).nunique(), pdf.a.groupby(pdkey).nunique(), check_names=False)
        assert_eq(ddf.a.groupby(ddkey).first(), pdf.a.groupby(pdkey).first(), check_names=False)
        assert_eq(ddf.a.groupby(ddkey).last(), pdf.a.groupby(pdkey).last(), check_names=False)
        assert_eq(ddf.a.groupby(ddkey).prod(), pdf.a.groupby(pdkey).prod(), check_names=False)
        for ddof in ddofs:
            assert_eq(ddf.a.groupby(ddkey).var(ddof), pdf.a.groupby(pdkey).var(ddof))
            assert_eq(ddf.a.groupby(ddkey).std(ddof), pdf.a.groupby(pdkey).std(ddof))
    for i in [0, 4, 7]:
        assert_eq(ddf.groupby(ddf.b > i).a.sum(), pdf.groupby(pdf.b > i).a.sum())
        assert_eq(ddf.groupby(ddf.b > i).a.min(), pdf.groupby(pdf.b > i).a.min())
        assert_eq(ddf.groupby(ddf.b > i).a.max(), pdf.groupby(pdf.b > i).a.max())
        assert_eq(ddf.groupby(ddf.b > i).a.count(), pdf.groupby(pdf.b > i).a.count())
        assert_eq(ddf.groupby(ddf.b > i).a.mean(), pdf.groupby(pdf.b > i).a.mean())
        assert_eq(ddf.groupby(ddf.b > i).a.nunique(), pdf.groupby(pdf.b > i).a.nunique())
        assert_eq(ddf.groupby(ddf.b > i).a.size(), pdf.groupby(pdf.b > i).a.size())
        assert_eq(ddf.groupby(ddf.b > i).a.first(), pdf.groupby(pdf.b > i).a.first())
        assert_eq(ddf.groupby(ddf.b > i).a.last(), pdf.groupby(pdf.b > i).a.last())
        assert_eq(ddf.groupby(ddf.b > i).a.tail(), pdf.groupby(pdf.b > i).a.tail())
        assert_eq(ddf.groupby(ddf.b > i).a.head(), pdf.groupby(pdf.b > i).a.head())
        assert_eq(ddf.groupby(ddf.b > i).a.prod(), pdf.groupby(pdf.b > i).a.prod())
        assert_eq(ddf.groupby(ddf.a > i).b.sum(), pdf.groupby(pdf.a > i).b.sum())
        assert_eq(ddf.groupby(ddf.a > i).b.min(), pdf.groupby(pdf.a > i).b.min())
        assert_eq(ddf.groupby(ddf.a > i).b.max(), pdf.groupby(pdf.a > i).b.max())
        assert_eq(ddf.groupby(ddf.a > i).b.count(), pdf.groupby(pdf.a > i).b.count())
        assert_eq(ddf.groupby(ddf.a > i).b.mean(), pdf.groupby(pdf.a > i).b.mean())
        assert_eq(ddf.groupby(ddf.a > i).b.nunique(), pdf.groupby(pdf.a > i).b.nunique())
        assert_eq(ddf.groupby(ddf.b > i).b.size(), pdf.groupby(pdf.b > i).b.size())
        assert_eq(ddf.groupby(ddf.b > i).b.first(), pdf.groupby(pdf.b > i).b.first())
        assert_eq(ddf.groupby(ddf.b > i).b.last(), pdf.groupby(pdf.b > i).b.last())
        assert_eq(ddf.groupby(ddf.b > i).b.tail(), pdf.groupby(pdf.b > i).b.tail())
        assert_eq(ddf.groupby(ddf.b > i).b.head(), pdf.groupby(pdf.b > i).b.head())
        assert_eq(ddf.groupby(ddf.b > i).b.prod(), pdf.groupby(pdf.b > i).b.prod())
        assert_eq(ddf.groupby(ddf.b > i).sum(), pdf.groupby(pdf.b > i).sum())
        assert_eq(ddf.groupby(ddf.b > i).min(), pdf.groupby(pdf.b > i).min())
        assert_eq(ddf.groupby(ddf.b > i).max(), pdf.groupby(pdf.b > i).max())
        assert_eq(ddf.groupby(ddf.b > i).count(), pdf.groupby(pdf.b > i).count())
        assert_eq(ddf.groupby(ddf.b > i).mean(), pdf.groupby(pdf.b > i).mean())
        assert_eq(ddf.groupby(ddf.b > i).size(), pdf.groupby(pdf.b > i).size())
        assert_eq(ddf.groupby(ddf.b > i).first(), pdf.groupby(pdf.b > i).first())
        assert_eq(ddf.groupby(ddf.b > i).last(), pdf.groupby(pdf.b > i).last())
        assert_eq(ddf.groupby(ddf.b > i).prod(), pdf.groupby(pdf.b > i).prod())
        assert_eq(ddf.groupby(ddf.a > i).sum(), pdf.groupby(pdf.a > i).sum())
        assert_eq(ddf.groupby(ddf.a > i).min(), pdf.groupby(pdf.a > i).min())
        assert_eq(ddf.groupby(ddf.a > i).max(), pdf.groupby(pdf.a > i).max())
        assert_eq(ddf.groupby(ddf.a > i).count(), pdf.groupby(pdf.a > i).count())
        assert_eq(ddf.groupby(ddf.a > i).mean(), pdf.groupby(pdf.a > i).mean())
        assert_eq(ddf.groupby(ddf.a > i).size(), pdf.groupby(pdf.a > i).size())
        assert_eq(ddf.groupby(ddf.a > i).first(), pdf.groupby(pdf.a > i).first())
        assert_eq(ddf.groupby(ddf.a > i).last(), pdf.groupby(pdf.a > i).last())
        assert_eq(ddf.groupby(ddf.a > i).prod(), pdf.groupby(pdf.a > i).prod())
        for ddof in ddofs:
            assert_eq(ddf.groupby(ddf.b > i).std(ddof), pdf.groupby(pdf.b > i).std(ddof))
    for ddkey, pdkey in [('a', 'a'), (ddf.a, pdf.a), (ddf.a + 1, pdf.a + 1), (ddf.a > 3, pdf.a > 3)]:
        assert_eq(ddf.groupby(ddkey).b.sum(), pdf.groupby(pdkey).b.sum())
        assert_eq(ddf.groupby(ddkey).b.min(), pdf.groupby(pdkey).b.min())
        assert_eq(ddf.groupby(ddkey).b.max(), pdf.groupby(pdkey).b.max())
        assert_eq(ddf.groupby(ddkey).b.count(), pdf.groupby(pdkey).b.count())
        assert_eq(ddf.groupby(ddkey).b.mean(), pdf.groupby(pdkey).b.mean())
        assert_eq(ddf.groupby(ddkey).b.nunique(), pdf.groupby(pdkey).b.nunique())
        assert_eq(ddf.groupby(ddkey).b.size(), pdf.groupby(pdkey).b.size())
        assert_eq(ddf.groupby(ddkey).b.first(), pdf.groupby(pdkey).b.first())
        assert_eq(ddf.groupby(ddkey).last(), pdf.groupby(pdkey).last())
        assert_eq(ddf.groupby(ddkey).prod(), pdf.groupby(pdkey).prod())
        assert_eq(ddf.groupby(ddkey).sum(), pdf.groupby(pdkey).sum())
        assert_eq(ddf.groupby(ddkey).min(), pdf.groupby(pdkey).min())
        assert_eq(ddf.groupby(ddkey).max(), pdf.groupby(pdkey).max())
        assert_eq(ddf.groupby(ddkey).count(), pdf.groupby(pdkey).count())
        assert_eq(ddf.groupby(ddkey).mean(), pdf.groupby(pdkey).mean().astype(float))
        assert_eq(ddf.groupby(ddkey).size(), pdf.groupby(pdkey).size())
        assert_eq(ddf.groupby(ddkey).first(), pdf.groupby(pdkey).first())
        assert_eq(ddf.groupby(ddkey).last(), pdf.groupby(pdkey).last())
        assert_eq(ddf.groupby(ddkey).prod(), pdf.groupby(pdkey).prod())
        for ddof in ddofs:
            assert_eq(ddf.groupby(ddkey).b.std(ddof), pdf.groupby(pdkey).b.std(ddof))
    assert sorted(ddf.groupby('b').a.sum().dask) == sorted(ddf.groupby('b').a.sum().dask)
    assert sorted(ddf.groupby(ddf.a > 3).b.mean().dask) == sorted(ddf.groupby(ddf.a > 3).b.mean().dask)
    pytest.raises(KeyError, lambda: ddf.groupby('x'))
    pytest.raises(KeyError, lambda: ddf.groupby(['a', 'x']))
    pytest.raises(KeyError, lambda: ddf.groupby('a')['x'])
    pytest.raises(KeyError, lambda: ddf.groupby('a')['b', 'x'])
    pytest.raises(KeyError, lambda: ddf.groupby('a')[['b', 'x']])
    if not DASK_EXPR_ENABLED:
        assert_dask_graph(ddf.groupby('b').a.sum(), 'series-groupby-sum')
        assert_dask_graph(ddf.groupby('b').a.min(), 'series-groupby-min')
        assert_dask_graph(ddf.groupby('b').a.max(), 'series-groupby-max')
        assert_dask_graph(ddf.groupby('b').a.count(), 'series-groupby-count')
        assert_dask_graph(ddf.groupby('b').a.var(), 'series-groupby-var')
        assert_dask_graph(ddf.groupby('b').a.cov(), 'series-groupby-cov')
        assert_dask_graph(ddf.groupby('b').a.first(), 'series-groupby-first')
        assert_dask_graph(ddf.groupby('b').a.last(), 'series-groupby-last')
        assert_dask_graph(ddf.groupby('b').a.tail(), 'series-groupby-tail')
        assert_dask_graph(ddf.groupby('b').a.head(), 'series-groupby-head')
        assert_dask_graph(ddf.groupby('b').a.prod(), 'series-groupby-prod')
        assert_dask_graph(ddf.groupby('b').a.mean(), 'series-groupby-sum')
        assert_dask_graph(ddf.groupby('b').a.mean(), 'series-groupby-count')
        assert_dask_graph(ddf.groupby('b').a.nunique(), 'series-groupby-nunique')
        assert_dask_graph(ddf.groupby('b').a.size(), 'series-groupby-size')
        assert_dask_graph(ddf.groupby('b').sum(), 'dataframe-groupby-sum')
        assert_dask_graph(ddf.groupby('b').min(), 'dataframe-groupby-min')
        assert_dask_graph(ddf.groupby('b').max(), 'dataframe-groupby-max')
        assert_dask_graph(ddf.groupby('b').count(), 'dataframe-groupby-count')
        assert_dask_graph(ddf.groupby('b').first(), 'dataframe-groupby-first')
        assert_dask_graph(ddf.groupby('b').last(), 'dataframe-groupby-last')
        assert_dask_graph(ddf.groupby('b').prod(), 'dataframe-groupby-prod')
        assert_dask_graph(ddf.groupby('b').mean(), 'dataframe-groupby-sum')
        assert_dask_graph(ddf.groupby('b').mean(), 'dataframe-groupby-count')
        assert_dask_graph(ddf.groupby('b').size(), 'dataframe-groupby-size')