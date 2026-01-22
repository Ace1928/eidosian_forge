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
@pytest.mark.parametrize('data', [test_data['int_data']], ids=['int_data'])
def test_pivot_table_dropna(data):
    eval_general(*create_test_dfs(data), operation=lambda df, *args, **kwargs: df.pivot_table(*args, **kwargs), index=lambda df: df.columns[0], columns=lambda df: df.columns[1], values=lambda df: df.columns[-1], dropna=False)