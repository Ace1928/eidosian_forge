import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.testing import assert_series_equal
from modin.tests.pandas.utils import (
@pytest.mark.parametrize('numeric_only', [False, True])
def test_count_specific(numeric_only):
    eval_general(*create_test_dfs(test_data_diff_dtype), lambda df: df.count(numeric_only=numeric_only))