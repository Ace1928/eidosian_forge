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
def test___setattr__mutating_column():
    pandas_df = pandas.DataFrame([[1]], columns=['col0'])
    modin_df = pd.DataFrame([[1]], columns=['col0'])
    pandas_df.col0 = [3]
    modin_df.col0 = [3]
    df_equals(modin_df, pandas_df)
    df_equals(modin_df.col0, pandas_df.col0)
    pandas_df.col0 = pandas.Series([5])
    modin_df.col0 = pd.Series([5])
    df_equals(modin_df, pandas_df)
    pandas_df.loc[0, 'col0'] = 4
    modin_df.loc[0, 'col0'] = 4
    df_equals(modin_df, pandas_df)
    assert modin_df.col0.equals(modin_df['col0'])
    with pytest.warns(UserWarning, match=SET_DATAFRAME_ATTRIBUTE_WARNING):
        modin_df.col1 = [4]
    with warnings.catch_warnings():
        warnings.filterwarnings(action='error', message=SET_DATAFRAME_ATTRIBUTE_WARNING)
        modin_df.col1 = [5]
        modin_df.new_attr = 6
        modin_df.col0 = 7
    assert 'new_attr' in dir(modin_df), 'Modin attribute was not correctly added to the df.'
    assert 'new_attr' not in modin_df, 'New attribute was not correctly added to columns.'
    assert modin_df.new_attr == 6, 'Modin attribute value was set incorrectly.'
    assert isinstance(modin_df.col0, pd.Series), 'Scalar was not broadcasted properly to an existing column.'