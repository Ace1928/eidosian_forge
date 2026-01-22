import io
import warnings
import matplotlib
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
@pytest.mark.parametrize('axis', axis_values, ids=axis_keys)
@pytest.mark.parametrize('skipna', [False, True])
@pytest.mark.parametrize('numeric_only', [False, True])
@pytest.mark.parametrize('method', ['kurtosis', 'kurt'])
def test_kurt_kurtosis(axis, skipna, numeric_only, method):
    data = test_data['float_nan_data']
    eval_general(*create_test_dfs(data), lambda df: getattr(df, method)(axis=axis, skipna=skipna, numeric_only=numeric_only))