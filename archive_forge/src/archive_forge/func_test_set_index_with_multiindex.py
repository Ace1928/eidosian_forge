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
@pytest.mark.parametrize('index', ['a', ['a', ('b', '')]])
def test_set_index_with_multiindex(index):
    kwargs = {'columns': [['a', 'b', 'c', 'd'], ['', '', 'x', 'y']]}
    modin_df, pandas_df = create_test_dfs(np.random.rand(2, 4), **kwargs)
    eval_general(modin_df, pandas_df, lambda df: df.set_index(index))