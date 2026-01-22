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
def test_loc_7135():
    data = np.random.randint(0, 100, size=(2 ** 16, 2 ** 8))
    modin_df, pandas_df = create_test_dfs(data)
    key = len(pandas_df)
    eval_loc(modin_df, pandas_df, value=list(range(2 ** 8)), key=key)