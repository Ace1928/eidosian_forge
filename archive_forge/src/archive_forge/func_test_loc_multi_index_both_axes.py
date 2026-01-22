import os
import sys
import matplotlib
import numpy as np
import pandas
import pytest
from pandas._testing import ensure_clean
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions, StorageFormat
from modin.pandas.indexing import is_range_like
from modin.pandas.testing import assert_index_equal
from modin.tests.pandas.utils import (
from modin.utils import get_current_execution
def test_loc_multi_index_both_axes():
    multi_index = pd.MultiIndex.from_tuples([('r0', 'rA'), ('r1', 'rB')], names=['Courses', 'Fee'])
    cols = pd.MultiIndex.from_tuples([('Gasoline', 'Toyota'), ('Gasoline', 'Ford'), ('Electric', 'Tesla'), ('Electric', 'Nio')])
    data = [[100, 300, 900, 400], [200, 500, 300, 600]]
    modin_df, pandas_df = create_test_dfs(data, columns=cols, index=multi_index)
    eval_general(modin_df, pandas_df, lambda df: df.loc[('r0', 'rA'), :])
    eval_general(modin_df, pandas_df, lambda df: df.loc[:, ('Gasoline', 'Toyota')])