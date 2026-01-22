import io
import warnings
import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import NPartitions
from modin.pandas.utils import SET_DATAFRAME_ATTRIBUTE_WARNING
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
def test___contains__(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    result = False
    key = 'Not Exist'
    assert result == modin_df.__contains__(key)
    assert result == (key in modin_df)
    if 'empty_data' not in request.node.name:
        result = True
        key = pandas_df.columns[0]
        assert result == modin_df.__contains__(key)
        assert result == (key in modin_df)