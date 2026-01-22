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
def test_to_list_of_structs_pandas(self):
    ints = pa.array([1, 2, 3], pa.int32())
    strings = pa.array([['a', 'b'], ['c', 'd'], ['e', 'f']], pa.list_(pa.string()))
    structs = pa.StructArray.from_arrays([ints, strings], ['f1', 'f2'])
    data = pa.ListArray.from_arrays([0, 1, 3], structs)
    expected = pd.Series([[{'f1': 1, 'f2': ['a', 'b']}], [{'f1': 2, 'f2': ['c', 'd']}, {'f1': 3, 'f2': ['e', 'f']}]])
    series = pd.Series(data.to_pandas())
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'elementwise comparison failed', DeprecationWarning)
        tm.assert_series_equal(series, expected)