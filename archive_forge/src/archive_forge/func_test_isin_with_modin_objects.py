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
def test_isin_with_modin_objects():
    modin_df1, pandas_df1 = create_test_dfs({'a': [1, 2], 'b': [3, 4]})
    modin_series, pandas_series = (pd.Series([1, 4, 5, 6]), pandas.Series([1, 4, 5, 6]))
    eval_general((modin_df1, modin_series), (pandas_df1, pandas_series), lambda srs: srs[0].isin(srs[1]))
    modin_df2 = modin_series.to_frame('a')
    pandas_df2 = pandas_series.to_frame('a')
    eval_general((modin_df1, modin_df2), (pandas_df1, pandas_df2), lambda srs: srs[0].isin(srs[1]))
    modin_df1, pandas_df1 = create_test_dfs({'a': [1, 2], 'b': [3, 4]}, index=[10, 11])
    eval_general((modin_df1, modin_series), (pandas_df1, pandas_series), lambda srs: srs[0].isin(srs[1]))
    eval_general((modin_df1, modin_df2), (pandas_df1, pandas_df2), lambda srs: srs[0].isin(srs[1]))