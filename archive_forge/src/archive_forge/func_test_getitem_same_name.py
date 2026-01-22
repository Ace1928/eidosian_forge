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
def test_getitem_same_name():
    data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
    columns = ['c1', 'c2', 'c1', 'c3']
    modin_df = pd.DataFrame(data, columns=columns)
    pandas_df = pandas.DataFrame(data, columns=columns)
    df_equals(modin_df['c1'], pandas_df['c1'])
    df_equals(modin_df['c2'], pandas_df['c2'])
    df_equals(modin_df[['c1', 'c2']], pandas_df[['c1', 'c2']])
    df_equals(modin_df['c3'], pandas_df['c3'])