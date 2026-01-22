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
@pytest.mark.parametrize('center', [True, False])
@pytest.mark.parametrize('closed', ['right', 'left', 'both', 'neither'])
@pytest.mark.parametrize('as_index', [True, False])
@pytest.mark.parametrize('on', [None, 'col4'])
def test_rolling_timedelta_window(center, closed, as_index, on):
    col_part1 = pd.DataFrame({'by': np.tile(np.arange(15), 10), 'col1': np.arange(150), 'col2': np.arange(10, 160)})
    col_part2 = pd.DataFrame({'col3': np.arange(20, 170)})
    if on is not None:
        col_part2[on] = pandas.DatetimeIndex([datetime.date(2020, 1, 1) + datetime.timedelta(hours=12) * i for i in range(150)])
    md_df = pd.concat([col_part1, col_part2], axis=1)
    md_df.index = pandas.DatetimeIndex([datetime.date(2020, 1, 1) + datetime.timedelta(days=1) * i for i in range(150)])
    pd_df = md_df._to_pandas()
    if StorageFormat.get() == 'Pandas':
        assert md_df._query_compiler._modin_frame._partitions.shape[1] == 2 if on is None else 3
    md_window = md_df.groupby('by', as_index=as_index).rolling(datetime.timedelta(days=3), center=center, closed=closed, on=on)
    pd_window = pd_df.groupby('by', as_index=as_index).rolling(datetime.timedelta(days=3), center=center, closed=closed, on=on)
    eval_rolling(md_window, pd_window)