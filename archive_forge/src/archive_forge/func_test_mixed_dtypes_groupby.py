import datetime
import itertools
from unittest import mock
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
import modin.pandas as pd
from modin.config import (
from modin.core.dataframe.algebra.default2pandas.groupby import GroupBy
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.pandas.io import from_pandas
from modin.pandas.utils import is_scalar
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import (
from .utils import (
@pytest.mark.parametrize('as_index', [True, False])
def test_mixed_dtypes_groupby(as_index):
    frame_data = np.random.RandomState(42).randint(97, 198, size=(2 ** 6, 2 ** 4))
    pandas_df = pandas.DataFrame(frame_data).add_prefix('col')
    for col in pandas_df.iloc[:, [i for i in range(len(pandas_df.columns)) if i % 2 == 0]]:
        pandas_df[col] = [str(chr(i)) for i in pandas_df[col]]
    modin_df = from_pandas(pandas_df)
    n = 1
    by_values = [('col1',), (lambda x: x % 2,), (modin_df['col0'].copy(), pandas_df['col0'].copy()), ('col3',)]
    for by in by_values:
        if isinstance(by[0], str) and by[0] == 'col3':
            modin_groupby = modin_df.set_index(by[0]).groupby(by=by[0], as_index=as_index)
            pandas_groupby = pandas_df.set_index(by[0]).groupby(by=by[-1], as_index=as_index)
            md_sorted_grpby = modin_df.set_index(by[0]).sort_index().groupby(by=by[0], as_index=as_index)
            pd_sorted_grpby = pandas_df.set_index(by[0]).sort_index().groupby(by=by[0], as_index=as_index)
        else:
            modin_groupby = modin_df.groupby(by=by[0], as_index=as_index)
            pandas_groupby = pandas_df.groupby(by=by[-1], as_index=as_index)
            md_sorted_grpby, pd_sorted_grpby = (modin_groupby, pandas_groupby)
        modin_groupby_equals_pandas(modin_groupby, pandas_groupby)
        eval_ngroups(modin_groupby, pandas_groupby)
        eval_general(md_sorted_grpby, pd_sorted_grpby, lambda df: df.ffill(), comparator=lambda *dfs: df_equals(*sort_if_experimental_groupby(*dfs)))
        eval_general(modin_groupby, pandas_groupby, lambda df: df.sem(), modin_df_almost_equals_pandas, expected_exception=False)
        eval_general(modin_groupby, pandas_groupby, lambda df: df.sample(random_state=1))
        eval_general(modin_groupby, pandas_groupby, lambda df: df.ewm(com=0.5).std(), expected_exception=pandas.errors.DataError('Cannot aggregate non-numeric type: object'))
        eval_shift(modin_groupby, pandas_groupby, comparator=assert_set_of_rows_identical if use_range_partitioning_groupby() else None)
        eval_mean(modin_groupby, pandas_groupby, numeric_only=True)
        eval_any(modin_groupby, pandas_groupby)
        eval_min(modin_groupby, pandas_groupby)
        eval_general(modin_groupby, pandas_groupby, lambda df: df.idxmax())
        eval_ndim(modin_groupby, pandas_groupby)
        eval_cumsum(modin_groupby, pandas_groupby, numeric_only=True)
        eval_general(modin_groupby, pandas_groupby, lambda df: df.pct_change(), modin_df_almost_equals_pandas, expected_exception=False)
        eval_cummax(modin_groupby, pandas_groupby, numeric_only=True)
        apply_functions = [lambda df: df.sum(), min]
        for func in apply_functions:
            eval_apply(modin_groupby, pandas_groupby, func)
        eval_dtypes(modin_groupby, pandas_groupby)
        eval_general(modin_groupby, pandas_groupby, lambda df: df.first(), comparator=lambda *dfs: df_equals(*sort_if_experimental_groupby(*dfs)))
        eval_cummin(modin_groupby, pandas_groupby, numeric_only=True)
        eval_general(md_sorted_grpby, pd_sorted_grpby, lambda df: df.bfill(), comparator=lambda *dfs: df_equals(*sort_if_experimental_groupby(*dfs)))
        eval_general(modin_groupby, pandas_groupby, lambda df: df.idxmin(numeric_only=True))
        eval_prod(modin_groupby, pandas_groupby, numeric_only=True)
        if as_index:
            eval_std(modin_groupby, pandas_groupby, numeric_only=True)
            eval_var(modin_groupby, pandas_groupby, numeric_only=True)
            eval_skew(modin_groupby, pandas_groupby, numeric_only=True)
        agg_functions = [lambda df: df.sum(), 'min', min, 'max', max, sum, {'col2': 'sum'}, {'col2': sum}, {'col2': 'max', 'col4': 'sum', 'col5': 'min'}, {'col2': max, 'col4': sum, 'col5': 'min'}, {'col0': 'count', 'col1': 'count', 'col2': 'count'}, {'col0': 'nunique', 'col1': 'nunique', 'col2': 'nunique'}]
        for func in agg_functions:
            eval_agg(modin_groupby, pandas_groupby, func)
        eval_general(modin_groupby, pandas_groupby, lambda df: df.last())
        eval_max(modin_groupby, pandas_groupby)
        eval_len(modin_groupby, pandas_groupby)
        eval_sum(modin_groupby, pandas_groupby)
        if not use_range_partitioning_groupby():
            eval_ngroup(modin_groupby, pandas_groupby)
        eval_nunique(modin_groupby, pandas_groupby)
        eval_value_counts(modin_groupby, pandas_groupby)
        eval_median(modin_groupby, pandas_groupby, numeric_only=True)
        eval_general(modin_groupby, pandas_groupby, lambda df: df.head(n), comparator=lambda *dfs: df_equals(*sort_if_experimental_groupby(*dfs)))
        eval_cumprod(modin_groupby, pandas_groupby, numeric_only=True)
        eval_general(modin_groupby, pandas_groupby, lambda df: df.cov(numeric_only=True), modin_df_almost_equals_pandas)
        transform_functions = [lambda df: df, lambda df: df + df]
        for func in transform_functions:
            eval_transform(modin_groupby, pandas_groupby, func)
        pipe_functions = [lambda dfgb: dfgb.sum()]
        for func in pipe_functions:
            eval_pipe(modin_groupby, pandas_groupby, func)
        eval_general(modin_groupby, pandas_groupby, lambda df: df.corr(numeric_only=True), modin_df_almost_equals_pandas)
        eval_fillna(modin_groupby, pandas_groupby)
        eval_count(modin_groupby, pandas_groupby)
        eval_general(modin_groupby, pandas_groupby, lambda df: df.tail(n), comparator=lambda *dfs: df_equals(*sort_if_experimental_groupby(*dfs)))
        eval_quantile(modin_groupby, pandas_groupby)
        eval_general(modin_groupby, pandas_groupby, lambda df: df.take([0]))
        eval___getattr__(modin_groupby, pandas_groupby, 'col2')
        eval_groups(modin_groupby, pandas_groupby)