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
def test_iloc_assignment():
    modin_df = pd.DataFrame(index=['row1', 'row2', 'row3'], columns=['col1', 'col2'])
    pandas_df = pandas.DataFrame(index=['row1', 'row2', 'row3'], columns=['col1', 'col2'])
    modin_df.iloc[0]['col1'] = 11
    modin_df.iloc[1]['col1'] = 21
    modin_df.iloc[2]['col1'] = 31
    modin_df.iloc[lambda df: 0]['col2'] = 12
    modin_df.iloc[1][lambda df: ['col2']] = 22
    modin_df.iloc[lambda df: 2][lambda df: ['col2']] = 32
    pandas_df.iloc[0]['col1'] = 11
    pandas_df.iloc[1]['col1'] = 21
    pandas_df.iloc[2]['col1'] = 31
    pandas_df.iloc[lambda df: 0]['col2'] = 12
    pandas_df.iloc[1][lambda df: ['col2']] = 22
    pandas_df.iloc[lambda df: 2][lambda df: ['col2']] = 32
    df_equals(modin_df, pandas_df)