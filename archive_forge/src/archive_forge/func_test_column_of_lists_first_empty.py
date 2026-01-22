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
def test_column_of_lists_first_empty(self):
    num_lists = [[], [2, 3, 4], [3, 6, 7, 8], [], [2]]
    series = pd.Series([np.array(s, dtype=float) for s in num_lists])
    arr = pa.array(series)
    result = pd.Series(arr.to_pandas())
    tm.assert_series_equal(result, series)