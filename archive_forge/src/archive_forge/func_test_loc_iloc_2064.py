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
@pytest.mark.parametrize('locator_name', ['iloc', 'loc'])
def test_loc_iloc_2064(locator_name):
    modin_df, pandas_df = create_test_dfs(columns=['col1', 'col2'])
    if locator_name == 'iloc':
        expected_exception = IndexError('index 1 is out of bounds for axis 0 with size 0')
    else:
        _type = 'int32' if os.name == 'nt' else 'int64'
        expected_exception = KeyError(f"None of [Index([1], dtype='{_type}')] are in the [index]")
    eval_general(modin_df, pandas_df, lambda df: getattr(df, locator_name).__setitem__([1], [11, 22]), __inplace__=True, expected_exception=expected_exception)