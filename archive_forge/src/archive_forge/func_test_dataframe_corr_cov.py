import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from .utils import (
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
@pytest.mark.parametrize('min_periods', [None, 5])
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('method', ['corr', 'cov'])
def test_dataframe_corr_cov(data, min_periods, axis, method):
    with warns_that_defaulting_to_pandas():
        eval_general(*create_test_dfs(data), lambda df: getattr(df.expanding(min_periods=min_periods, axis=axis), method)())