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
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
def test_info_default_param(data):
    with io.StringIO() as first, io.StringIO() as second:
        eval_general(pd.DataFrame(data), pandas.DataFrame(data), verbose=None, max_cols=None, memory_usage=None, operation=lambda df, **kwargs: df.info(**kwargs), buf=lambda df: second if isinstance(df, pandas.DataFrame) else first)
        modin_info = first.getvalue().splitlines()
        pandas_info = second.getvalue().splitlines()
        assert modin_info[0] == str(pd.DataFrame)
        assert pandas_info[0] == str(pandas.DataFrame)
        assert modin_info[1:] == pandas_info[1:]