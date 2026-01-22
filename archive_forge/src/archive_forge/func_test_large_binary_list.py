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
def test_large_binary_list(self):
    for list_type_factory in (pa.list_, pa.large_list):
        s = pa.array([['aa', 'bb'], None, ['cc'], []], type=list_type_factory(pa.large_binary())).to_pandas()
        tm.assert_series_equal(s, pd.Series([[b'aa', b'bb'], None, [b'cc'], []]), check_names=False)
        s = pa.array([['aa', 'bb'], None, ['cc'], []], type=list_type_factory(pa.large_string())).to_pandas()
        tm.assert_series_equal(s, pd.Series([['aa', 'bb'], None, ['cc'], []]), check_names=False)