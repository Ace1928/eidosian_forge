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
@pytest.mark.parametrize('known_cats', [True, False], ids=['known', 'unknown'])
@pytest.mark.parametrize('ordered_cats', [True, False], ids=['ordered', 'unordererd'])
@pytest.mark.parametrize('groupby', ['cat_1', ['cat_1', 'cat_2']])
@pytest.mark.parametrize('observed', [True, False], ids=['observed', 'unobserved'])
def test_groupby_aggregate_categorical_observed(known_cats, ordered_cats, agg_func, groupby, observed):
    if agg_func in ['cov', 'corr', 'nunique']:
        pytest.skip('Not implemented for DataFrameGroupBy yet.')
    if agg_func == 'median' and isinstance(groupby, str):
        pytest.skip("Can't calculate median over categorical")
    if agg_func == 'median' and DASK_EXPR_ENABLED:
        pytest.skip("Can't deal with unobserved cats in median at the moment")
    if agg_func in ['sum', 'count', 'prod'] and groupby != 'cat_1':
        pytest.skip('Gives zeros rather than nans.')
    if agg_func in ['std', 'var'] and observed:
        pytest.skip("Can't calculate observed with all nans")
    if agg_func in ['sum', 'prod'] and PANDAS_GE_200:
        pytest.xfail('Not implemented for category type with pandas 2.0')
    pdf = pd.DataFrame({'cat_1': pd.Categorical(list('AB'), categories=list('ABCDE'), ordered=ordered_cats), 'cat_2': pd.Categorical([1, 2], categories=[1, 2, 3], ordered=ordered_cats), 'value_1': np.random.uniform(size=2)})
    ddf = dd.from_pandas(pdf, 2)
    if not known_cats:
        ddf['cat_1'] = ddf['cat_1'].cat.as_unknown()
        ddf['cat_2'] = ddf['cat_2'].cat.as_unknown()

    def agg(grp, **kwargs):
        if isinstance(grp, pd.core.groupby.DataFrameGroupBy) or (PANDAS_GE_150 and (not PANDAS_GE_200)):
            ctx = check_numeric_only_deprecation
        else:
            ctx = contextlib.nullcontext
        with ctx():
            return getattr(grp, agg_func)(**kwargs)
    if ordered_cats is False and agg_func in ['min', 'max'] and (groupby == 'cat_1'):
        pdf = pdf[['cat_1', 'value_1']]
        ddf = ddf[['cat_1', 'value_1']]
    assert_eq(agg(pdf.groupby(groupby, observed=observed)), agg(ddf.groupby(groupby, observed=observed)))