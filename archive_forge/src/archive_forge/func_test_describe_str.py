import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.testing import assert_series_equal
from modin.tests.pandas.utils import (
@pytest.mark.parametrize('data', [test_data['int_data']])
def test_describe_str(data):
    modin_df = pd.DataFrame(data).applymap(str)
    pandas_df = pandas.DataFrame(data).applymap(str)
    try:
        df_equals(modin_df.describe(), pandas_df.describe())
    except AssertionError:
        df_equals(modin_df.describe().loc[['count', 'unique', 'freq']], pandas_df.describe().loc[['count', 'unique', 'freq']])