import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.tests.pandas.utils import (
def test_diff_with_datetime_types():
    pandas_df = pandas.DataFrame([[1, 2.0, 3], [4, 5.0, 6], [7, np.nan, 9], [10, 11.3, 12], [13, 14.5, 15]])
    data = pandas.date_range('2018-01-01', periods=5, freq='h').values
    pandas_df = pandas.concat([pandas_df, pandas.Series(data)], axis=1)
    modin_df = pd.DataFrame(pandas_df)
    pandas_result = pandas_df.diff()
    modin_result = modin_df.diff()
    df_equals(modin_result, pandas_result)
    td_pandas_result = pandas_result.diff()
    td_modin_result = modin_result.diff()
    df_equals(td_modin_result, td_pandas_result)