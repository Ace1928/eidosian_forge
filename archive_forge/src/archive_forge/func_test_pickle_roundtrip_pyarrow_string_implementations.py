from __future__ import annotations
import pickle
from datetime import date, datetime, time, timedelta
from decimal import Decimal
import numpy as np
import pandas as pd
import pandas._testing as tm
import pytest
from dask.dataframe.utils import get_string_dtype
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_150
@pytest.mark.parametrize('string_dtype', ['stringdtype', pytest.param('arrowdtype', marks=pytest.mark.skipif(not PANDAS_GE_150, reason='Requires ArrowDtype'))])
def test_pickle_roundtrip_pyarrow_string_implementations(string_dtype):
    if string_dtype == 'stringdtype':
        string_dtype = pd.StringDtype('pyarrow')
    else:
        string_dtype = pd.ArrowDtype(pa.string())
    expected = pd.Series(map(str, range(1000)), dtype=string_dtype)
    expected_sliced = expected.head(2)
    full_pickled = pickle.dumps(expected)
    sliced_pickled = pickle.dumps(expected_sliced)
    assert len(full_pickled) > len(sliced_pickled) * 3
    result = pickle.loads(full_pickled)
    tm.assert_series_equal(result, expected)
    result_sliced = pickle.loads(sliced_pickled)
    tm.assert_series_equal(result_sliced, expected_sliced)