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
def test_category_implicit_from_pandas(self):

    def _check(v):
        arr = pa.array(v)
        result = arr.to_pandas()
        tm.assert_series_equal(pd.Series(result), pd.Series(v))
    arrays = [pd.Categorical(['a', 'b', 'c'], categories=['a', 'b']), pd.Categorical(['a', 'b', 'c'], categories=['a', 'b'], ordered=True)]
    for arr in arrays:
        _check(arr)