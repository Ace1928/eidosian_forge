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
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
def test_series_to_timedelta(data):

    def make_frame(lib):
        series = lib.Series(next(iter(data.values())) if isinstance(data, dict) else data)
        return lib.to_timedelta(series).to_frame(name='timedelta')
    eval_general(pd, pandas, make_frame)