from __future__ import annotations
import datetime
import itertools
import json
import unittest.mock as mock
import matplotlib
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal
from pandas.core.indexing import IndexingError
from pandas.errors import SpecificationError
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.pandas.testing import assert_series_equal
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution, try_cast_to_pandas
from .utils import (
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
@pytest.mark.parametrize('bound_type', ['list', 'series'], ids=['list', 'series'])
def test_clip_sequence(request, data, bound_type):
    modin_series, pandas_series = create_test_series(data)
    if name_contains(request.node.name, numeric_dfs):
        lower = random_state.randint(RAND_LOW, RAND_HIGH, len(pandas_series))
        upper = random_state.randint(RAND_LOW, RAND_HIGH, len(pandas_series))
        if bound_type == 'series':
            modin_lower = pd.Series(lower)
            pandas_lower = pandas.Series(lower)
            modin_upper = pd.Series(upper)
            pandas_upper = pandas.Series(upper)
        else:
            modin_lower = pandas_lower = lower
            modin_upper = pandas_upper = upper
        modin_result = modin_series.clip(modin_lower, modin_upper, axis=0)
        pandas_result = pandas_series.clip(pandas_lower, pandas_upper)
        df_equals(modin_result, pandas_result)
        modin_result = modin_series.clip(np.nan, modin_upper, axis=0)
        pandas_result = pandas_series.clip(np.nan, pandas_upper)
        df_equals(modin_result, pandas_result)