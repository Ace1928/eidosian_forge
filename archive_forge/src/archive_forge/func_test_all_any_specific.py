import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.testing import assert_series_equal
from modin.tests.pandas.utils import (
@pytest.mark.parametrize('method', ['all', 'any'])
@pytest.mark.parametrize('bool_only', bool_arg_values, ids=arg_keys('bool_only', bool_arg_keys))
def test_all_any_specific(bool_only, method):
    eval_general(*create_test_dfs(test_data_diff_dtype), lambda df: getattr(df, method)(bool_only=bool_only))