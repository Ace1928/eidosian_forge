import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.tests.pandas.utils import (
@pytest.mark.parametrize('skipna', [False, True])
@pytest.mark.parametrize('ddof', int_arg_values, ids=arg_keys('ddof', int_arg_keys))
def test_sem_float_nan_only(skipna, ddof):
    eval_general(*create_test_dfs(test_data['float_nan_data']), lambda df: df.sem(skipna=skipna, ddof=ddof))