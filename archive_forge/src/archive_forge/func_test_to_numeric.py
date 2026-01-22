import contextlib
import numpy as np
import pandas
import pytest
from numpy.testing import assert_array_equal
import modin.pandas as pd
from modin.config import StorageFormat
from modin.pandas.io import to_pandas
from modin.pandas.testing import assert_frame_equal
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
from .utils import (
@pytest.mark.parametrize('data, errors, downcast', [(['1.0', '2', -3], 'raise', None), (['1.0', '2', -3], 'raise', 'float'), (['1.0', '2', -3], 'raise', 'signed'), (['apple', '1.0', '2', -3], 'ignore', None), (['apple', '1.0', '2', -3], 'coerce', None)])
def test_to_numeric(data, errors, downcast):
    modin_series = pd.Series(data)
    pandas_series = pandas.Series(data)
    modin_result = pd.to_numeric(modin_series, errors=errors, downcast=downcast)
    pandas_result = pandas.to_numeric(pandas_series, errors=errors, downcast=downcast)
    df_equals(modin_result, pandas_result)