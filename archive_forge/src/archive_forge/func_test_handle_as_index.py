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
@pytest.mark.parametrize('internal_by_length', [0, 1, 2])
@pytest.mark.parametrize('external_by_length', [0, 1, 2])
@pytest.mark.parametrize('has_categorical_by', [True, False])
@pytest.mark.parametrize('agg_func', [pytest.param(lambda grp: grp.apply(lambda df: df.dtypes), id='modin_dtypes_impl'), pytest.param(lambda grp: grp.apply(lambda df: df.sum(numeric_only=True)), id='apply_sum'), pytest.param(lambda grp: grp.count(), id='count'), pytest.param(lambda grp: grp.nunique(), id='nunique'), pytest.param({1: 'sum', 2: 'nunique'}, id='dict_agg_no_intersection_with_by'), pytest.param({0: 'mean', 1: 'sum', 2: 'nunique'}, id='dict_agg_has_intersection_with_by'), pytest.param({1: 'sum', 2: 'nunique', -1: 'nunique'}, id='dict_agg_has_intersection_with_categorical_by')])
@pytest.mark.parametrize('use_backend_agnostic_method', [True, False])
def test_handle_as_index(internal_by_length, external_by_length, has_categorical_by, agg_func, use_backend_agnostic_method, request):
    """
    Test ``modin.core.dataframe.algebra.default2pandas.groupby.GroupBy.handle_as_index``.

    The role of the ``handle_as_index`` method is to build a groupby result considering
    ``as_index=False`` from the result that was computed with ``as_index=True``.

    So the testing flow is the following:
        1. Compute GroupBy result with the ``as_index=True`` parameter via Modin.
        2. Build ``as_index=False`` result from the ``as_index=True`` using ``handle_as_index`` method.
        3. Compute GroupBy result with the ``as_index=False`` parameter via pandas as the reference result.
        4. Compare the result from the second step with the reference.
    """
    by_length = internal_by_length + external_by_length
    if by_length == 0:
        pytest.skip('No keys to group on were passed, skipping the test.')
    if has_categorical_by and by_length > 1 and (isinstance(agg_func, dict) or 'nunique' in request.node.callspec.id.split('-')):
        pytest.skip("The linked bug makes pandas raise an exception when 'by' is categorical: " + 'https://github.com/pandas-dev/pandas/issues/36698')
    df = pandas.DataFrame(test_groupby_data)
    external_by_cols = GroupBy.validate_by(df.add_prefix('external_'))
    if has_categorical_by:
        df = df.astype({df.columns[-1]: 'category'})
    if isinstance(agg_func, dict):
        agg_func = {df.columns[key]: value for key, value in agg_func.items()}
        selection = list(agg_func.keys())
        agg_dict = agg_func
        agg_func = lambda grp: grp.agg(agg_dict)
    else:
        selection = None
    internal_by = df.columns[range(-internal_by_length // 2, internal_by_length // 2)].tolist()
    external_by = external_by_cols[:external_by_length]
    pd_by = internal_by + external_by
    md_by = internal_by + [pd.Series(ser) for ser in external_by]
    grp_result = pd.DataFrame(df).groupby(md_by, as_index=True)
    grp_reference = df.groupby(pd_by, as_index=False)
    agg_result = agg_func(grp_result)
    agg_reference = agg_func(grp_reference)
    if use_backend_agnostic_method:
        reset_index, drop, lvls_to_drop, cols_to_drop = GroupBy.handle_as_index(result_cols=agg_result.columns, result_index_names=agg_result.index.names, internal_by_cols=internal_by, by_cols_dtypes=df[internal_by].dtypes.values, by_length=len(md_by), selection=selection, drop=len(internal_by) != 0)
        if len(lvls_to_drop) > 0:
            agg_result.index = agg_result.index.droplevel(lvls_to_drop)
        if len(cols_to_drop) > 0:
            agg_result = agg_result.drop(columns=cols_to_drop)
        if reset_index:
            agg_result = agg_result.reset_index(drop=drop)
    else:
        GroupBy.handle_as_index_for_dataframe(result=agg_result, internal_by_cols=internal_by, by_cols_dtypes=df[internal_by].dtypes.values, by_length=len(md_by), selection=selection, drop=len(internal_by) != 0, inplace=True)
    df_equals(agg_result, agg_reference)