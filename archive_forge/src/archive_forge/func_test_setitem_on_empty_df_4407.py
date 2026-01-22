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
def test_setitem_on_empty_df_4407():
    data = {}
    index = pd.date_range(end='1/1/2018', periods=0, freq='D')
    column = pd.date_range(end='1/1/2018', periods=1, freq='h')[0]
    modin_df = pd.DataFrame(data, columns=index)
    pandas_df = pandas.DataFrame(data, columns=index)
    modin_df[column] = pd.Series([1])
    pandas_df[column] = pandas.Series([1])
    df_equals(modin_df, pandas_df)
    assert modin_df.columns.freq == pandas_df.columns.freq