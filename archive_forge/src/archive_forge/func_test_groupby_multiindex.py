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
@pytest.mark.parametrize('groupby_kwargs', [pytest.param({'level': 1, 'axis': 1}, id='level_idx_axis=1'), pytest.param({'level': 1}, id='level_idx'), pytest.param({'level': [1, 'four']}, id='level_idx+name'), pytest.param({'by': 'four'}, id='level_name'), pytest.param({'by': ['one', 'two']}, id='level_name_multi_by'), pytest.param({'by': ['item0', 'one', 'two']}, id='col_name+level_name')])
def test_groupby_multiindex(groupby_kwargs):
    frame_data = np.random.randint(0, 100, size=(2 ** 6, 2 ** 6))
    modin_df = pd.DataFrame(frame_data)
    pandas_df = pandas.DataFrame(frame_data)
    new_index = pandas.Index([f'item{i}' for i in range(len(pandas_df))])
    new_columns = pandas.MultiIndex.from_tuples([(i // 4, i // 2, i) for i in modin_df.columns], names=['four', 'two', 'one'])
    modin_df.columns = new_columns
    modin_df.index = new_index
    pandas_df.columns = new_columns
    pandas_df.index = new_index
    if groupby_kwargs.get('axis', 0) == 0:
        modin_df = modin_df.T
        pandas_df = pandas_df.T
    md_grp, pd_grp = (modin_df.groupby(**groupby_kwargs), pandas_df.groupby(**groupby_kwargs))
    modin_groupby_equals_pandas(md_grp, pd_grp)
    df_equals(md_grp.sum(), pd_grp.sum())
    df_equals(md_grp.size(), pd_grp.size())
    df_equals(md_grp.first(), pd_grp.first())