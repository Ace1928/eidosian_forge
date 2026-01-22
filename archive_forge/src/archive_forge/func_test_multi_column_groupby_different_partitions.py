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
@pytest.mark.parametrize('func_to_apply', [lambda df: df.sum(), lambda df: df.size(), lambda df: df.quantile(), lambda df: df.dtypes, lambda df: df.apply(lambda df: df.sum()), pytest.param(lambda df: df.apply(lambda df: pandas.Series([1, 2, 3, 4])), marks=pytest.mark.skip('See modin issue #2511')), lambda grp: grp.agg({list(test_data_values[0].keys())[1]: (max, min, sum), list(test_data_values[0].keys())[-2]: (sum, min, max)}), lambda grp: grp.agg({list(test_data_values[0].keys())[1]: [('new_sum', 'sum'), ('new_min', 'min')], list(test_data_values[0].keys())[-2]: np.sum}), pytest.param(lambda grp: grp.agg({list(test_data_values[0].keys())[1]: [('new_sum', 'sum'), ('new_mean', 'mean')], list(test_data_values[0].keys())[-2]: 'skew'}), id='renaming_aggs_at_different_partitions'), pytest.param(lambda grp: grp.agg({list(test_data_values[0].keys())[1]: [('new_sum', 'sum'), ('new_mean', 'mean')], list(test_data_values[0].keys())[2]: 'skew'}), id='renaming_aggs_at_same_partition'), pytest.param(lambda grp: grp.agg({list(test_data_values[0].keys())[1]: 'mean', list(test_data_values[0].keys())[-2]: 'skew'}), id='custom_aggs_at_different_partitions'), pytest.param(lambda grp: grp.agg({list(test_data_values[0].keys())[1]: 'mean', list(test_data_values[0].keys())[2]: 'skew'}), id='custom_aggs_at_same_partition'), pytest.param(lambda grp: grp.agg({list(test_data_values[0].keys())[1]: 'mean', list(test_data_values[0].keys())[-2]: 'sum'}), id='native_and_custom_aggs_at_different_partitions'), pytest.param(lambda grp: grp.agg({list(test_data_values[0].keys())[1]: 'mean', list(test_data_values[0].keys())[2]: 'sum'}), id='native_and_custom_aggs_at_same_partition'), pytest.param(lambda grp: grp.agg({list(test_data_values[0].keys())[1]: (max, 'mean', sum), list(test_data_values[0].keys())[-1]: (sum, 'skew', max)}), id='Agg_and_by_intersection_TreeReduce_implementation'), pytest.param(lambda grp: grp.agg({list(test_data_values[0].keys())[1]: (max, 'mean', 'nunique'), list(test_data_values[0].keys())[-1]: (sum, min, max)}), id='Agg_and_by_intersection_FullAxis_implementation'), pytest.param(lambda grp: grp.agg({list(test_data_values[0].keys())[0]: 'count'}), id='Agg_and_by_intersection_issue_3376')])
@pytest.mark.parametrize('as_index', [True, False])
@pytest.mark.parametrize('by_length', [1, 2])
@pytest.mark.parametrize('categorical_by', [pytest.param(True, marks=pytest.mark.skip('See modin issue #2513')), False])
def test_multi_column_groupby_different_partitions(func_to_apply, as_index, by_length, categorical_by, request):
    if not categorical_by and by_length == 1 and ('custom_aggs_at_same_partition' in request.node.name) or 'renaming_aggs_at_same_partition' in request.node.name:
        pytest.xfail('After upgrade to pandas 2.1 skew results are different: AssertionError: 1.0 >= 0.0001.' + ' See https://github.com/modin-project/modin/issues/6530 for details.')
    data = test_data_values[0]
    md_df, pd_df = create_test_dfs(data)
    by = [pd_df.columns[-i if i % 2 else i] for i in range(by_length)]
    if categorical_by:
        md_df = md_df.astype({by[0]: 'category'})
        pd_df = pd_df.astype({by[0]: 'category'})
    md_grp, pd_grp = (md_df.groupby(by, as_index=as_index), pd_df.groupby(by, as_index=as_index))
    eval_general(md_grp, pd_grp, func_to_apply, comparator=try_modin_df_almost_equals_compare)
    eval___getitem__(md_grp, pd_grp, md_df.columns[1], expected_exception=False)
    eval___getitem__(md_grp, pd_grp, [md_df.columns[1], md_df.columns[2]], expected_exception=False)