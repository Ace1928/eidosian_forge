import warnings
import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions, RangePartitioning, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('ascending', [False, True])
@pytest.mark.parametrize('na_position', ['first', 'last'], ids=['first', 'last'])
def test_sort_index(axis, ascending, na_position):
    data = test_data['float_nan_data']
    modin_df, pandas_df = (pd.DataFrame(data), pandas.DataFrame(data))
    if axis == 0:
        length = len(modin_df.index)
        for df in [modin_df, pandas_df]:
            df.index = [(i - length / 2) % length for i in range(length)]
    dfs = [modin_df, pandas_df]
    for idx in range(len(dfs)):
        sort_index = dfs[idx].axes[axis]
        dfs[idx] = dfs[idx].set_axis([np.nan if i % 2 == 0 else sort_index[i] for i in range(len(sort_index))], axis=axis, copy=False)
    modin_df, pandas_df = dfs
    eval_general(modin_df, pandas_df, lambda df: df.sort_index(axis=axis, ascending=ascending, na_position=na_position))