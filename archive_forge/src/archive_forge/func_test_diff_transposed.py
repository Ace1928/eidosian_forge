import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.tests.pandas.utils import (
@pytest.mark.parametrize('axis', ['rows', 'columns'])
def test_diff_transposed(axis):
    eval_general(*create_test_dfs(test_data['int_data']), lambda df: df.T.diff(axis=axis))