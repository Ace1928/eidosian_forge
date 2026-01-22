import gc
import decimal
import json
import multiprocessing as mp
import sys
import warnings
from collections import OrderedDict
from datetime import date, datetime, time, timedelta, timezone
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest
from pyarrow.pandas_compat import get_logical_type, _pandas_api
from pyarrow.tests.util import invoke_script, random_ascii, rands
import pyarrow.tests.strategies as past
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
import pyarrow as pa
def test_to_list_of_maps_pandas_sliced(self):
    """
        A slightly more rigorous test for chunk/slice combinations
        """
    if Version(np.__version__) >= Version('1.25.0.dev0') and Version(pd.__version__) < Version('2.0.0'):
        pytest.skip('Regression in pandas with numpy 1.25')
    keys = pa.array(['ignore', 'foo', 'bar', 'baz', 'qux', 'quux', 'ignore']).slice(1, 5)
    items = pa.array([['ignore'], ['ignore'], ['a', 'b'], ['c', 'd'], [], None, [None, 'e']], pa.list_(pa.string())).slice(2, 5)
    map = pa.MapArray.from_arrays([0, 2, 4], keys, items)
    arr = pa.ListArray.from_arrays([0, 1, 2], map)
    series = arr.to_pandas()
    expected = pd.Series([[[('foo', ['a', 'b']), ('bar', ['c', 'd'])]], [[('baz', []), ('qux', None)]]])
    series_sliced = arr.slice(1, 2).to_pandas()
    expected_sliced = pd.Series([[[('baz', []), ('qux', None)]]])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'elementwise comparison failed', DeprecationWarning)
        tm.assert_series_equal(series, expected)
        tm.assert_series_equal(series_sliced, expected_sliced)