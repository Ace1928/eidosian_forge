import io
import warnings
import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions
from modin.pandas.utils import SET_DATAFRAME_ATTRIBUTE_WARNING
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
def test_inplace_series_ops(data):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)
    if len(modin_df.columns) > len(pandas_df.columns):
        col0 = modin_df.columns[0]
        col1 = modin_df.columns[1]
        pandas_df[col1].dropna(inplace=True)
        modin_df[col1].dropna(inplace=True)
        df_equals(modin_df, pandas_df)
        pandas_df[col0].fillna(0, inplace=True)
        modin_df[col0].fillna(0, inplace=True)
        df_equals(modin_df, pandas_df)