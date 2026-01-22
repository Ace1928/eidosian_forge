import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.testing import assert_series_equal
from modin.tests.pandas.utils import (
def test_describe_dtypes():
    data = {'col1': list('abc'), 'col2': list('abc'), 'col3': list('abc'), 'col4': [1, 2, 3]}
    eval_general(*create_test_dfs(data), lambda df: df.describe())