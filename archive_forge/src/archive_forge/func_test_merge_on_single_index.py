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
@pytest.mark.parametrize('left_index', [[], ['key'], ['key', 'b'], ['key', 'b', 'c'], ['b'], ['b', 'c']])
@pytest.mark.parametrize('right_index', [[], ['key'], ['key', 'e'], ['key', 'e', 'f'], ['e'], ['e', 'f']])
def test_merge_on_single_index(left_index, right_index):
    """
    Test ``.merge()`` method when merging on a single column, that is located in an index level of one of the frames.
    """
    modin_df1, pandas_df1 = create_test_dfs({'b': [3, 4, 4, 5], 'key': [1, 1, 2, 2], 'c': [2, 3, 2, 2], 'd': [2, 1, 3, 1]})
    if len(left_index):
        modin_df1 = modin_df1.set_index(left_index)
        pandas_df1 = pandas_df1.set_index(left_index)
    modin_df2, pandas_df2 = create_test_dfs({'e': [3, 4, 4, 5], 'f': [2, 3, 2, 2], 'key': [1, 1, 2, 2], 'h': [2, 1, 3, 1]})
    if len(right_index):
        modin_df2 = modin_df2.set_index(right_index)
        pandas_df2 = pandas_df2.set_index(right_index)
    eval_general((modin_df1, modin_df2), (pandas_df1, pandas_df2), lambda dfs: dfs[0].merge(dfs[1], on='key'))