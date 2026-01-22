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
def test_array_from_pandas_date_with_mask(self):
    m = np.array([True, False, True])
    data = pd.Series([date(1990, 1, 1), date(1991, 1, 1), date(1992, 1, 1)])
    result = pa.Array.from_pandas(data, mask=m)
    expected = pd.Series([None, date(1991, 1, 1), None])
    assert pa.Array.from_pandas(expected).equals(result)