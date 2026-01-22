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
@pytest.mark.parametrize('as_index', [True, pytest.param(False, marks=pytest.mark.skipif(get_current_execution() == 'BaseOnPython' or use_range_partitioning_groupby(), reason='See Pandas issue #39103'))])
@pytest.mark.parametrize('by_length', [1, 3])
@pytest.mark.parametrize('agg_fns', [['sum', 'min', 'max'], ['mean', 'quantile']], ids=['reduce', 'aggregation'])
@pytest.mark.parametrize('intersection_with_by_cols', [pytest.param(True, marks=pytest.mark.skip('See Modin issue #3602')), False])
def test_dict_agg_rename_mi_columns(as_index, by_length, agg_fns, intersection_with_by_cols):
    md_df, pd_df = create_test_dfs(test_data['int_data'])
    mi_columns = generate_multiindex(len(md_df.columns), nlevels=4)
    md_df.columns, pd_df.columns = (mi_columns, mi_columns)
    by = list(md_df.columns[:by_length])
    agg_cols = list(md_df.columns[by_length - 1:by_length + 2]) if intersection_with_by_cols else list(md_df.columns[by_length:by_length + 3])
    agg_dict = {f'custom-{i}' + str(agg_fns[i % len(agg_fns)]): (col, agg_fns[i % len(agg_fns)]) for i, col in enumerate(agg_cols)}
    md_res = md_df.groupby(by, as_index=as_index).agg(**agg_dict)
    pd_res = pd_df.groupby(by, as_index=as_index).agg(**agg_dict)
    df_equals(md_res, pd_res)