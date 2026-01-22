import contextlib
import numpy as np
import pandas
import pytest
from numpy.testing import assert_array_equal
import modin.pandas as pd
from modin.config import StorageFormat
from modin.pandas.io import to_pandas
from modin.pandas.testing import assert_frame_equal
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
from .utils import (
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
@pytest.mark.parametrize('append_na', [True, False])
@pytest.mark.parametrize('op', ['isna', 'isnull', 'notna', 'notnull'])
def test_isna_isnull_notna_notnull(data, append_na, op):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(pandas_df)
    if append_na:
        pandas_df['NONE_COL'] = None
        pandas_df['NAN_COL'] = np.nan
        modin_df['NONE_COL'] = None
        modin_df['NAN_COL'] = np.nan
    pandas_result = getattr(pandas, op)(pandas_df)
    modin_result = getattr(pd, op)(modin_df)
    df_equals(modin_result, pandas_result)
    modin_result = getattr(pd, op)(pd.Series([1, np.nan, 2]))
    pandas_result = getattr(pandas, op)(pandas.Series([1, np.nan, 2]))
    df_equals(modin_result, pandas_result)
    assert pd.isna(np.nan) == pandas.isna(np.nan)