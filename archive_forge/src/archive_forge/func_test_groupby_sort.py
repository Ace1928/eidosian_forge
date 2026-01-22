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
@pytest.mark.parametrize('sort', [True, False])
@pytest.mark.parametrize('is_categorical_by', [True, False])
def test_groupby_sort(sort, is_categorical_by):
    by = np.array(['a'] * 50000 + ['b'] * 10000 + ['c'] * 1000)
    random_state = np.random.RandomState(seed=42)
    random_state.shuffle(by)
    data = {'key_col': by, 'data_col': np.arange(len(by))}
    md_df, pd_df = create_test_dfs(data)
    if is_categorical_by:
        md_df = md_df.astype({'key_col': 'category'})
        pd_df = pd_df.astype({'key_col': 'category'})
    md_grp = md_df.groupby('key_col', sort=sort)
    pd_grp = pd_df.groupby('key_col', sort=sort)
    modin_groupby_equals_pandas(md_grp, pd_grp)
    eval_general(md_grp, pd_grp, lambda grp: grp.sum(numeric_only=True))
    eval_general(md_grp, pd_grp, lambda grp: grp.size())
    eval_general(md_grp, pd_grp, lambda grp: grp.agg(lambda df: df.mean()))
    eval_general(md_grp, pd_grp, lambda grp: grp.dtypes)
    eval_general(md_grp, pd_grp, lambda grp: grp.first())