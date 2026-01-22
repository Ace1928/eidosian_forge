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
@pytest.mark.parametrize('modin_df_recipe', ['non_lazy_frame', 'frame_with_deferred_index', 'lazy_frame'])
def test_groupby_on_empty_data(modin_df_recipe):

    class ModinDfConstructor:

        def __init__(self, recipe, df_kwargs):
            self._recipe = recipe
            self._mock_obj = None
            self._df_kwargs = df_kwargs

        def non_lazy_frame(self):
            return pd.DataFrame(**self._df_kwargs)

        def frame_with_deferred_index(self):
            df = pd.DataFrame(**self._df_kwargs)
            try:
                df._query_compiler._modin_frame.set_index_cache(None)
            except AttributeError:
                pytest.skip(reason="Selected execution doesn't support deferred indices.")
            return df

        def lazy_frame(self):
            donor_obj = pd.DataFrame()._query_compiler
            self._mock_obj = mock.patch(f'{donor_obj.__module__}.{donor_obj.__class__.__name__}.lazy_execution', new_callable=mock.PropertyMock)
            patch_obj = self._mock_obj.__enter__()
            patch_obj.return_value = True
            df = pd.DataFrame(**self._df_kwargs)
            assert df._query_compiler.lazy_execution
            return df

        def __enter__(self):
            return getattr(self, self._recipe)()

        def __exit__(self, *args, **kwargs):
            if self._mock_obj is not None:
                self._mock_obj.__exit__(*args, **kwargs)

    def run_test(eval_function, *args, **kwargs):
        df_kwargs = {'columns': ['a', 'b', 'c']}
        with ModinDfConstructor(modin_df_recipe, df_kwargs) as modin_df:
            pandas_df = pandas.DataFrame(**df_kwargs)
            modin_grp = modin_df.groupby(modin_df.columns[0])
            pandas_grp = pandas_df.groupby(pandas_df.columns[0])
            eval_function(modin_grp, pandas_grp, *args, **kwargs)
    run_test(eval___getattr__, item='b')
    run_test(eval___getitem__, item='b')
    run_test(eval_agg, func=lambda df: df.mean())
    run_test(eval_any)
    run_test(eval_apply, func=lambda df: df.mean())
    run_test(eval_count)
    run_test(eval_cummax, numeric_only=True)
    run_test(eval_cummin, numeric_only=True)
    run_test(eval_cumprod, numeric_only=True)
    run_test(eval_cumsum, numeric_only=True)
    run_test(eval_dtypes)
    run_test(eval_fillna)
    run_test(eval_groups)
    run_test(eval_len)
    run_test(eval_max)
    run_test(eval_mean)
    run_test(eval_median)
    run_test(eval_min)
    run_test(eval_ndim)
    run_test(eval_ngroup)
    run_test(eval_ngroups)
    run_test(eval_nunique)
    run_test(eval_prod)
    run_test(eval_quantile)
    run_test(eval_rank)
    run_test(eval_size)
    run_test(eval_skew)
    run_test(eval_sum)
    run_test(eval_var)
    if modin_df_recipe != 'lazy_frame':
        run_test(eval_pipe, func=lambda df: df.mean())
        run_test(eval_shift)