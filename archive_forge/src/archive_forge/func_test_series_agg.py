import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from .utils import (
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
@pytest.mark.parametrize('min_periods', [None, 5])
def test_series_agg(data, min_periods):
    modin_series, pandas_series = create_test_series(data)
    pandas_expanded = pandas_series.expanding(min_periods=min_periods)
    modin_expanded = modin_series.expanding(min_periods=min_periods)
    df_equals(modin_expanded.aggregate(np.sum), pandas_expanded.aggregate(np.sum))
    df_equals(pandas_expanded.aggregate([np.sum, np.mean]), modin_expanded.aggregate([np.sum, np.mean]))