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
def test_loc_multi_index_duplicate_keys():
    modin_df, pandas_df = create_test_dfs([1, 2], index=[['a', 'a'], ['b', 'b']])
    eval_general(modin_df, pandas_df, lambda df: df.loc[('a', 'b'), 0])
    eval_general(modin_df, pandas_df, lambda df: df.loc[('a', 'b'), :])