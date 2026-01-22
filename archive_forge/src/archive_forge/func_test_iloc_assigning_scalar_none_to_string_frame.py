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
def test_iloc_assigning_scalar_none_to_string_frame():
    data = [['A']]
    modin_df = pd.DataFrame(data, dtype='string')
    modin_df.iloc[0, 0] = None
    pandas_df = pandas.DataFrame(data, dtype='string')
    pandas_df.iloc[0, 0] = None
    df_equals(modin_df, pandas_df)