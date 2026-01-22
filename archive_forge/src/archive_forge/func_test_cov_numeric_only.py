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
@pytest.mark.parametrize('numeric_only', [True, False])
def test_cov_numeric_only(numeric_only):
    if not numeric_only:
        pytest.xfail(reason='https://github.com/modin-project/modin/issues/7023')
    eval_general(*create_test_dfs({'a': [1, 2, 3], 'b': [3, 2, 5], 'c': ['a', 'b', 'c']}), lambda df: df.cov(numeric_only=numeric_only))