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
def test_index_to_datetime_using_set_index():
    data = {'YEAR': ['1992', '1993', '1994'], 'ALIENS': [1, 99, 1]}
    modin_df_years = pd.DataFrame(data=data)
    df_years = pandas.DataFrame(data=data)
    modin_df_years = modin_df_years.set_index('YEAR')
    df_years = df_years.set_index('YEAR')
    modin_datetime_index = pd.to_datetime(modin_df_years.index, format='%Y')
    pandas_datetime_index = pandas.to_datetime(df_years.index, format='%Y')
    modin_df_years.index = modin_datetime_index
    df_years.index = pandas_datetime_index
    modin_df_years.set_index(modin_datetime_index)
    df_years.set_index(pandas_datetime_index)
    df_equals(modin_df_years, df_years)