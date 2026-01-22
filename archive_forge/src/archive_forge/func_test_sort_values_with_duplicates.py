import warnings
import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions, RangePartitioning, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
def test_sort_values_with_duplicates():
    modin_df = pd.DataFrame({'col': [2, 1, 1]}, index=[1, 1, 0])
    pandas_df = pandas.DataFrame({'col': [2, 1, 1]}, index=[1, 1, 0])
    key = modin_df.columns[0]
    modin_result = modin_df.sort_values(key, inplace=False)
    pandas_result = pandas_df.sort_values(key, inplace=False)
    df_equals(modin_result, pandas_result)
    modin_df.sort_values(key, inplace=True)
    pandas_df.sort_values(key, inplace=True)
    df_equals(modin_df, pandas_df)