import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.testing import assert_series_equal
from modin.tests.pandas.utils import (
@pytest.mark.parametrize('exclude,include', [([np.float64], None), (np.float64, None), (None, [np.timedelta64, np.datetime64, np.object_, np.bool_]), (None, 'all'), (None, np.number)])
def test_describe_specific(exclude, include):
    eval_general(*create_test_dfs(test_data_diff_dtype), lambda df: df.drop('str_col', axis=1).describe(exclude=exclude, include=include))